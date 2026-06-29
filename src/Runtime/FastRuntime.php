<?php

declare(strict_types=1);

namespace Rotifer\Runtime;

/**
 * Bakes the "fast run" PHP runtime in: OPcache + tracing JIT on, Xdebug off.
 *
 * These are load-time settings (a zend_extension and PHP_INI_SYSTEM ini values),
 * so they cannot be flipped with ini_set() once PHP is running - the only place to
 * apply them is the process launch. {@see accelerate()} therefore re-executes the
 * current bin/rotifer invocation once with the right -d flags; the dashboard's
 * RunManager spawns runs with the same {@see flags()} directly. Together that means
 * a plain `php bin/rotifer run ...` (or a run started from the dashboard) gets the
 * ~40-60x forward-pass speedup with no flags to remember.
 *
 * Re-exec is a no-op when it would not help (JIT already live) or when opted out
 * (ROTIFER_NO_FAST=1 / --no-fast - e.g. to keep Xdebug coverage on a run), and the
 * child carries a sentinel so it never re-execs again. Determinism is unaffected:
 * a JIT'd run produces the byte-identical champion of a non-JIT run.
 */
final class FastRuntime
{
    /** Set on the re-exec'd child so it does not re-exec a second time. */
    public const SENTINEL = 'ROTIFER_FAST';

    private const OPT_OUT = 'ROTIFER_NO_FAST';

    /**
     * Re-exec the current invocation under the fast runtime and exit with its
     * status. A no-op (returns normally) when acceleration is unnecessary, opted
     * out, or the child process cannot be launched - the caller then just runs in
     * the current, slower process.
     *
     * @param list<string> $argv the entry script's $argv ($argv[0] = script path)
     */
    public static function accelerate(array $argv): void
    {
        if (!self::shouldAccelerate($argv)) {
            return;
        }

        $command = array_merge([PHP_BINARY], self::flags(), [$argv[0]], array_slice($argv, 1));
        $process = @proc_open(
            $command,
            [STDIN, STDOUT, STDERR], // inherit the terminal so the live dashboard still renders
            $pipes,
            null,
            self::childEnv(),
            ['bypass_shell' => true],
        );
        if (!is_resource($process)) {
            return;
        }
        exit(proc_close($process));
    }

    /** Whether re-execing into the fast runtime would help and is allowed. */
    public static function shouldAccelerate(array $argv): bool
    {
        if (getenv(self::SENTINEL) !== false) {
            return false; // already the accelerated child
        }
        if (getenv(self::OPT_OUT) !== false || in_array('--no-fast', $argv, true)) {
            return false;
        }
        return !self::jitActive() || self::xdebugActive();
    }

    /**
     * The -d flags that turn OPcache + tracing JIT on and Xdebug off. Usable both
     * as argv elements (proc_open array form) and, escaped, in a shell command.
     *
     * @return list<string>
     */
    public static function flags(): array
    {
        $flags = [];
        // Only load OPcache ourselves when it is not already in - loading it twice errors.
        if (!extension_loaded('Zend OPcache') && ($opcache = self::locateOpcache()) !== null) {
            $flags[] = '-d';
            $flags[] = 'zend_extension=' . $opcache;
        }
        return array_merge($flags, [
            '-d', 'xdebug.mode=off',
            '-d', 'opcache.enable_cli=1',
            '-d', 'opcache.jit_buffer_size=64M',
            '-d', 'opcache.jit=tracing',
        ]);
    }

    /**
     * The current environment plus the sentinel that stops the child re-execing.
     * Pass this as proc_open's env so a spawned run inherits everything (PATH, etc.)
     * yet skips a redundant re-exec.
     *
     * @return array<string, string>
     */
    public static function childEnv(): array
    {
        $env = getenv();
        $env[self::SENTINEL] = '1';
        // Record whether the *base* ini already loads OPcache (visible here, before
        // the JIT re-exec adds it via -d), so worker inis don't try to load it twice.
        $env['ROTIFER_BASE_OPCACHE'] = extension_loaded('Zend OPcache') ? '1' : '0';
        return $env;
    }

    /**
     * Health of the *current* process: is the JIT live, and is Xdebug slowing it?
     * The accelerated run process calls this to self-report (the CLI prints it, the
     * run writes it into meta.json for the dashboard).
     *
     * @return array{fast: bool, jit: bool, opcache: bool, xdebug: bool, warnings: list<string>}
     */
    public static function diagnostics(): array
    {
        $jit = self::jitActive();
        $opcache = extension_loaded('Zend OPcache');
        $xdebug = self::xdebugActive();

        $warnings = [];
        if (!$jit) {
            $warnings[] = $opcache
                ? 'OPcache is loaded but its JIT is off - forward passes run several times slower. '
                  . 'Set opcache.jit=tracing and opcache.jit_buffer_size.'
                : 'OPcache/JIT is not active - forward passes run several times slower. '
                  . 'The OPcache extension could not be loaded.';
        }
        if ($xdebug) {
            $warnings[] = sprintf(
                'Xdebug is active (mode=%s) - it instruments every call and slows runs several times. '
                . 'Runs should set xdebug.mode=off.',
                ini_get('xdebug.mode') ?: '?',
            );
        }

        return [
            'fast' => $jit && !$xdebug,
            'jit' => $jit,
            'opcache' => $opcache,
            'xdebug' => $xdebug,
            'warnings' => $warnings,
        ];
    }

    /**
     * What a run launched from here *will* get. The dashboard server itself runs
     * un-accelerated, so its own {@see diagnostics()} would be misleading; this
     * instead predicts the spawned run's health. Runs always launch with Xdebug off,
     * so the only risk is OPcache not being installed where we can find it.
     *
     * @return array{fast: bool, jit: bool, warnings: list<string>}
     */
    public static function projectedRunHealth(): array
    {
        $opcacheAvailable = extension_loaded('Zend OPcache') || self::locateOpcache() !== null;

        $warnings = [];
        if (!$opcacheAvailable) {
            $warnings[] = sprintf(
                'OPcache extension not found in "%s" - runs cannot use JIT and will be several times slower. '
                . 'Install/enable OPcache for the full speed-up.',
                ini_get('extension_dir') ?: '(unknown)',
            );
        }

        return ['fast' => $opcacheAvailable, 'jit' => $opcacheAvailable, 'warnings' => $warnings];
    }

    /**
     * Ensure an ini fragment that turns OPcache JIT on (and Xdebug off) exists, and
     * return the directory holding it - for use as PHP_INI_SCAN_DIR when spawning
     * child PHP processes (e.g. amphp workers), which otherwise start from the base
     * php.ini with no JIT. Children inherit the env var, so they pick this up.
     */
    public static function workerIniDir(): string
    {
        $dir = sys_get_temp_dir() . DIRECTORY_SEPARATOR . 'rotifer-fast-ini';
        if (!is_dir($dir)) {
            @mkdir($dir, 0777, true);
        }
        $lines = [
            'opcache.enable=1',
            'opcache.enable_cli=1',
            'opcache.jit_buffer_size=64M',
            'opcache.jit=tracing',
            'xdebug.mode=off',
        ];
        // Workers start from the base ini (no -d flags), so they must load OPcache
        // themselves - but only when the base ini doesn't already (else a double-load
        // errors). childEnv() records the base state in ROTIFER_BASE_OPCACHE.
        if (getenv('ROTIFER_BASE_OPCACHE') === '0' && ($opcache = self::locateOpcache()) !== null) {
            array_unshift($lines, 'zend_extension=' . $opcache);
        }
        @file_put_contents($dir . DIRECTORY_SEPARATOR . 'zz-rotifer-fast.ini', implode("\n", $lines) . "\n");
        return $dir;
    }

    private static function jitActive(): bool
    {
        if (!function_exists('opcache_get_status')) {
            return false;
        }
        $status = @opcache_get_status(false);
        return is_array($status) && ($status['jit']['enabled'] ?? false) === true;
    }

    private static function xdebugActive(): bool
    {
        if (!extension_loaded('xdebug')) {
            return false;
        }
        $mode = (string) ini_get('xdebug.mode');
        return $mode !== '' && $mode !== 'off';
    }

    /** Absolute path to the OPcache extension in this PHP's extension dir, or null. */
    private static function locateOpcache(): ?string
    {
        $dir = (string) ini_get('extension_dir');
        if ($dir !== '' && !self::isAbsolute($dir)) {
            $dir = dirname(PHP_BINARY) . DIRECTORY_SEPARATOR . $dir;
        }
        $file = PHP_OS_FAMILY === 'Windows' ? 'php_opcache.dll' : 'opcache.so';
        $path = rtrim($dir, '/\\') . DIRECTORY_SEPARATOR . $file;
        return is_file($path) ? $path : null;
    }

    private static function isAbsolute(string $path): bool
    {
        return (bool) preg_match('#^([A-Za-z]:[\\\\/]|[\\\\/])#', $path);
    }
}