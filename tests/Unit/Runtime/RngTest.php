<?php

declare(strict_types=1);

namespace Rotifer\Tests\Unit\Runtime;

use PHPUnit\Framework\TestCase;
use Rotifer\Genome\Weight;
use Rotifer\Runtime\Rng;

final class RngTest extends TestCase
{
    public function testSameSeedProducesIdenticalSequence(): void
    {
        $a = new Rng(42);
        $b = new Rng(42);
        $seqA = array_map(fn () => $a->nextUint32(), range(1, 20));
        $seqB = array_map(fn () => $b->nextUint32(), range(1, 20));
        $this->assertSame($seqA, $seqB);
    }

    public function testDifferentSeedsDiverge(): void
    {
        $a = new Rng(1);
        $b = new Rng(2);
        $this->assertNotSame(
            array_map(fn () => $a->nextUint32(), range(1, 20)),
            array_map(fn () => $b->nextUint32(), range(1, 20)),
        );
    }

    public function testFloatStaysInUnitInterval(): void
    {
        $rng = new Rng(7);
        for ($i = 0; $i < 1000; $i++) {
            $f = $rng->nextFloat();
            $this->assertGreaterThanOrEqual(0.0, $f);
            $this->assertLessThan(1.0, $f);
        }
    }

    public function testIntBetweenIsInclusiveAndHitsBothEnds(): void
    {
        $rng = new Rng(123);
        $seen = [];
        for ($i = 0; $i < 2000; $i++) {
            $n = $rng->intBetween(3, 6);
            $this->assertGreaterThanOrEqual(3, $n);
            $this->assertLessThanOrEqual(6, $n);
            $seen[$n] = true;
        }
        ksort($seen);
        $this->assertSame([3, 4, 5, 6], array_keys($seen), 'every value in range is reachable, incl. both ends');
    }

    public function testChanceBoundaries(): void
    {
        $rng = new Rng(99);
        $this->assertFalse($rng->chance(0.0));
        $this->assertTrue($rng->chance(1.0));
    }

    public function testShuffleIsDeterministicAndPreservesElements(): void
    {
        $input = range(1, 10);
        $a = (new Rng(5))->shuffle($input);
        $b = (new Rng(5))->shuffle($input);
        $sorted = $a;
        sort($sorted);
        $this->assertSame($a, $b, 'shuffle is reproducible for a given seed');
        $this->assertSame($input, $sorted, 'no element is lost or duplicated');
    }

    public function testDeriveIsDeterministicAndIndependent(): void
    {
        $parent = new Rng(1000);
        $childA1 = $parent->derive(0);
        $childA2 = (new Rng(1000))->derive(0);
        $childB = $parent->derive(1);

        $this->assertSame($childA1->nextUint32(), $childA2->nextUint32(), 'derive is reproducible');
        $this->assertNotSame((new Rng(1000))->derive(0)->nextUint32(), $childB->nextUint32());
    }

    public function testWeightWithinLegalRange(): void
    {
        $rng = new Rng(3);
        for ($i = 0; $i < 500; $i++) {
            $this->assertTrue(Weight::isInRange($rng->weight()));
        }
    }
}
