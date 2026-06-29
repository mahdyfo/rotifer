<?php

declare(strict_types=1);

namespace Rotifer\Tests\Unit\Evolution;

use PHPUnit\Framework\TestCase;
use Rotifer\Evolution\Adaptation\AdaptiveMutation;

final class AdaptiveMutationTest extends TestCase
{
    public function testImprovingEasesMutationDown(): void
    {
        $adaptive = new AdaptiveMutation(patience: 3, upFactor: 2.0, downFactor: 0.5, minScale: 0.1, maxScale: 10.0);
        $this->assertSame(0.5, $adaptive->update(1.0));
        $this->assertSame(0.25, $adaptive->update(2.0));
    }

    public function testStagnationRampsMutationUpAfterPatience(): void
    {
        $adaptive = new AdaptiveMutation(patience: 2, upFactor: 2.0, downFactor: 0.5, minScale: 0.1, maxScale: 10.0);
        $adaptive->update(1.0);          // improve -> 0.5
        $adaptive->update(1.0);          // stall 1
        $scale = $adaptive->update(1.0); // stall 2 == patience -> *2 -> 1.0
        $this->assertSame(1.0, $scale);
    }

    public function testScaleStaysWithinBounds(): void
    {
        $adaptive = new AdaptiveMutation(patience: 1, upFactor: 5.0, downFactor: 0.1, minScale: 0.2, maxScale: 3.0);
        for ($i = 1; $i <= 20; $i++) {
            $adaptive->update((float) $i); // strictly increasing -> always improving -> floor
        }
        $this->assertSame(0.2, $adaptive->scale());

        for ($i = 0; $i < 20; $i++) {
            $adaptive->update(5.0); // no new best -> stagnation -> ramp to ceiling
        }
        $this->assertSame(3.0, $adaptive->scale());
    }
}
