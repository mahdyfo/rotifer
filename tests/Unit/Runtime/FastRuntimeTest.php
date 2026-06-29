<?php

declare(strict_types=1);

namespace Rotifer\Tests\Unit\Runtime;

use PHPUnit\Framework\TestCase;
use Rotifer\Runtime\FastRuntime;

final class FastRuntimeTest extends TestCase
{
    protected function setUp(): void
    {
        putenv('ROTIFER_FAST');
        putenv('ROTIFER_NO_FAST');
    }

    protected function tearDown(): void
    {
        putenv('ROTIFER_FAST');
        putenv('ROTIFER_NO_FAST');
    }

    public function testFlagsTurnJitOnAndXdebugOff(): void
    {
        $flags = implode(' ', FastRuntime::flags());

        $this->assertStringContainsString('xdebug.mode=off', $flags);
        $this->assertStringContainsString('opcache.enable_cli=1', $flags);
        $this->assertStringContainsString('opcache.jit=tracing', $flags);
        $this->assertStringContainsString('opcache.jit_buffer_size=64M', $flags);
    }

    public function testSkipsAccelerationOnceAlreadyInTheFastChild(): void
    {
        putenv('ROTIFER_FAST=1');

        $this->assertFalse(FastRuntime::shouldAccelerate(['bin/rotifer', 'run', 'xor']));
    }

    public function testOptsOutViaEnvVar(): void
    {
        putenv('ROTIFER_NO_FAST=1');

        $this->assertFalse(FastRuntime::shouldAccelerate(['bin/rotifer', 'run', 'xor']));
    }

    public function testOptsOutViaFlag(): void
    {
        $this->assertFalse(FastRuntime::shouldAccelerate(['bin/rotifer', 'run', 'xor', '--no-fast']));
    }

    public function testChildEnvCarriesTheSentinelAndPreservesExistingVars(): void
    {
        putenv('ROTIFER_TEST_MARKER=keep-me');
        try {
            $env = FastRuntime::childEnv();

            $this->assertSame('1', $env[FastRuntime::SENTINEL]);
            $this->assertSame('keep-me', $env['ROTIFER_TEST_MARKER']);
        } finally {
            putenv('ROTIFER_TEST_MARKER');
        }
    }
}