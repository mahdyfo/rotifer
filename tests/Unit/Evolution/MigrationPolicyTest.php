<?php

declare(strict_types=1);

namespace Rotifer\Tests\Unit\Evolution;

use PHPUnit\Framework\TestCase;
use Rotifer\Evolution\Migration\MigrationPolicy;

final class MigrationPolicyTest extends TestCase
{
    public function testDueOnTheScheduledInterval(): void
    {
        $policy = new MigrationPolicy(everyGenerations: 3, topK: 2);
        $this->assertFalse($policy->isDue(0));
        $this->assertFalse($policy->isDue(2));
        $this->assertTrue($policy->isDue(3));
        $this->assertTrue($policy->isDue(6));
    }

    public function testNeverDueWhenDisabled(): void
    {
        $policy = new MigrationPolicy(everyGenerations: 0, topK: 2);
        $this->assertFalse($policy->isDue(5));
    }

    public function testRingDestinationWrapsAround(): void
    {
        $policy = new MigrationPolicy(everyGenerations: 1, topK: 1);
        $this->assertSame(1, $policy->destinationOf(0, 4));
        $this->assertSame(0, $policy->destinationOf(3, 4));
        $this->assertSame(0, $policy->destinationOf(0, 1), 'a lone island sends to itself (no-op)');
    }
}
