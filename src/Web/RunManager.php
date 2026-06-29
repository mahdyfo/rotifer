<?php

declare(strict_types=1);

namespace Rotifer\Web;

use Rotifer\Persistence\SnapshotStore;
use Rotifer\Runtime\FastRuntime;

/**
 * Lets the dashboard launch and stop evolution runs. It shells out to the same
 * `bin/rotifer run ... --web` the CLI uses (so there is one code path), tracks
 * the active run in a small control file, and can kill it on request.
 *
 * Inputs come from HTTP, so the problem name is validated against the registry
 * and overrides are strictly whitelisted (numeric or boolean) and passed with
 * escaping - no arbitrary strings ever reach the shell.
 */
final class RunManager
{
    private const NUMERIC = ['seed', 'population', 'generations', 'islands', 'elitism', 'initial-hidden', 'migration-every', 'migration-top', 'simplicity'];
    private const FLOAT = ['crossover', 'weight-mutation', 'add-neuron', 'add-connection', 'remove-neuron', 'remove-connection', 'survive-rate', 'diversity'];
    private const BOOLEAN = ['trauma', 'adaptive-mutation', 'lifetime-learning'];
    private const ACTIVATIONS = ['sigmoid', 'relu', 'leaky_relu', 'tanh', 'threshold', 'gelu', 'softmax'];

    private readonly string $controlFile;

    public function __construct(
        private readonly string $projectRoot,
        private readonly SnapshotStore $store,
    ) {
        $this->controlFile = rtrim($projectRoot, '/\\') . '/runs/.control.json';
    }

    /**
     * @param array<string, scalar> $overrides
     * @return array{ok: bool, name?: string, error?: string}
     */
    public function start(string $problem, array $overrides): array
    {
        if (!preg_match('/^[a-z0-9_]+$/', $problem)) {
            return ['ok' => false, 'error' => 'invalid problem name'];
        }

        $this->stop(); // only one active run at a time

        $rotifer = $this->projectRoot . '/bin/rotifer';
        $args = ['run', $problem, '--web', '--quiet'];
        foreach (self::NUMERIC as $key) {
            if (isset($overrides[$key]) && is_numeric($overrides[$key])) {
                $args[] = '--' . $key . '=' . (int) $overrides[$key];
            }
        }
        foreach (self::FLOAT as $key) {
            if (isset($overrides[$key]) && is_numeric($overrides[$key])) {
                $args[] = '--' . $key . '=' . (float) $overrides[$key];
            }
        }
        foreach (self::BOOLEAN as $key) {
            if (array_key_exists($key, $overrides)) {
                $args[] = '--' . $key . '=' . ($this->truthy($overrides[$key]) ? '1' : '0');
            }
        }
        // Activation is a string from a fixed set (it reaches the shell, so whitelist it).
        if (isset($overrides['activation']) && in_array($overrides['activation'], self::ACTIVATIONS, true)) {
            $args[] = '--activation=' . $overrides['activation'];
        }
        // Hidden layers: a comma-separated list of sizes ("5,3,5") or empty for dynamic.
        // Only digits, commas and spaces reach the shell, and it is escaped below.
        if (isset($overrides['hidden-layers']) && is_string($overrides['hidden-layers'])
            && preg_match('/^[0-9, ]*$/', $overrides['hidden-layers'])) {
            $args[] = '--hidden-layers=' . $overrides['hidden-layers'];
        }
        // Parallel only when 2+ workers are asked for; 0/1 stays serial.
        if (isset($overrides['parallel']) && (int) $overrides['parallel'] >= 2) {
            $args[] = '--parallel=' . (int) $overrides['parallel'];
        }
        // Continue from the saved population instead of starting over.
        if (isset($overrides['resume']) && $this->truthy($overrides['resume'])) {
            $args[] = '--resume';
        }

        // Launch with OPcache JIT on and Xdebug off (the same speedup the CLI gets);
        // the sentinel in the spawn env stops bin/rotifer re-execing on top of this.
        $flags = implode(' ', array_map(escapeshellarg(...), FastRuntime::flags()));
        $command = escapeshellarg(PHP_BINARY) . ' ' . $flags . ' ' . escapeshellarg($rotifer) . ' '
            . implode(' ', array_map(escapeshellarg(...), $args));

        $this->store->ensure($problem);
        $logFile = $this->store->runDir($problem) . '/run.log';
        $pid = $this->spawnDetached($command, $logFile);
        if ($pid === null) {
            return ['ok' => false, 'error' => 'could not start process'];
        }

        $this->writeControl(['name' => $problem, 'problem' => $problem, 'pid' => $pid, 'startedAt' => date('c'), 'status' => 'running']);
        return ['ok' => true, 'name' => $problem];
    }

    public function stop(): void
    {
        $control = $this->readControl();
        if ($control === null) {
            return;
        }
        $pid = (int) ($control['pid'] ?? 0);
        if ($pid > 0 && $this->isAlive($pid)) {
            $this->kill($pid);
        }
        $control['status'] = 'stopped';
        $this->writeControl($control);
    }

    /**
     * @return array{active: ?string, running: bool, problem: ?string, runs: list<string>}
     */
    public function status(): array
    {
        $control = $this->readControl();
        $running = false;
        $active = null;
        $problem = null;
        if ($control !== null) {
            $active = $control['name'] ?? null;
            $problem = $control['problem'] ?? null;
            $running = ($control['status'] ?? '') === 'running'
                && isset($control['pid'])
                && $this->isAlive((int) $control['pid']);
            if (!$running && ($control['status'] ?? '') === 'running') {
                // process finished on its own; reflect that.
                $control['status'] = 'finished';
                $this->writeControl($control);
            }
        }

        $runs = $this->store->runs();
        $active ??= $runs[0] ?? null;

        return ['active' => $active, 'running' => $running, 'problem' => $problem, 'runs' => $runs];
    }

    private function spawnDetached(string $command, ?string $logFile = null): ?int
    {
        $isWindows = stripos(PHP_OS, 'WIN') === 0;
        $sink = $logFile ?? $this->nullDevice();
        $descriptors = [['pipe', 'r'], ['file', $sink, 'w'], ['file', $sink, 'a']];

        // create_process_group detaches from the server's Ctrl+C without opening a
        // visible console window (create_new_console did, which flashed on screen).
        $options = $isWindows ? ['bypass_shell' => true, 'create_process_group' => true] : [];
        $process = @proc_open($command, $descriptors, $pipes, $this->projectRoot, FastRuntime::childEnv(), $options);
        if (!is_resource($process)) {
            return null;
        }
        $status = proc_get_status($process);
        foreach ($pipes as $pipe) {
            if (is_resource($pipe)) {
                fclose($pipe);
            }
        }
        // Detach: do not proc_close (which would wait); the child keeps running.
        return $status['pid'] > 0 ? $status['pid'] : null;
    }

    private function isAlive(int $pid): bool
    {
        if (stripos(PHP_OS, 'WIN') === 0) {
            $out = shell_exec('tasklist /FI ' . escapeshellarg('PID eq ' . $pid) . ' /NH 2>NUL');
            return $out !== null && str_contains($out, (string) $pid);
        }
        return function_exists('posix_kill') ? posix_kill($pid, 0) : (shell_exec('ps -p ' . $pid) !== null);
    }

    private function kill(int $pid): void
    {
        if (stripos(PHP_OS, 'WIN') === 0) {
            shell_exec('taskkill /F /T /PID ' . $pid . ' 2>NUL');
        } else {
            shell_exec('kill -TERM ' . $pid . ' 2>/dev/null');
        }
    }

    private function nullDevice(): string
    {
        return stripos(PHP_OS, 'WIN') === 0 ? 'NUL' : '/dev/null';
    }

    private function truthy(mixed $value): bool
    {
        return filter_var($value, FILTER_VALIDATE_BOOLEAN, FILTER_NULL_ON_FAILURE) ?? (bool) $value;
    }

    private function readControl(): ?array
    {
        if (!is_file($this->controlFile)) {
            return null;
        }
        $data = json_decode((string) file_get_contents($this->controlFile), true);
        return is_array($data) ? $data : null;
    }

    private function writeControl(array $data): void
    {
        $dir = dirname($this->controlFile);
        if (!is_dir($dir)) {
            mkdir($dir, 0777, true);
        }
        file_put_contents($this->controlFile, json_encode($data, JSON_PRETTY_PRINT));
    }
}
