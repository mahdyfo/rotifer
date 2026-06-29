<?php

declare(strict_types=1);

namespace Rotifer\Tests\Unit\Organism;

use PHPUnit\Framework\TestCase;
use Rotifer\Organism\Epigenome;

final class EpigenomeTest extends TestCase
{
    public function testRaiseAccumulatesAndCapsAtOne(): void
    {
        $epi = new Epigenome();
        $epi->raise('stress', 0.6);
        $epi->raise('stress', 0.6);
        $this->assertSame(1.0, $epi->intensity('stress'));
    }

    public function testDecayFadesAndDropsNegligibleMarkers(): void
    {
        $epi = new Epigenome(['stress' => 0.4]);
        $epi->decay(0.5);
        $this->assertEqualsWithDelta(0.2, $epi->intensity('stress'), 1e-9);

        $epi->decay(0.01); // 0.002 -> below threshold -> removed
        $this->assertTrue($epi->isEmpty());
    }

    public function testInheritAveragesParentMarkers(): void
    {
        $child = Epigenome::inherit(
            new Epigenome(['stress' => 0.8]),
            new Epigenome(['stress' => 0.2, 'heat' => 0.4]),
        );
        $this->assertEqualsWithDelta(0.5, $child->intensity('stress'), 1e-9);
        $this->assertEqualsWithDelta(0.2, $child->intensity('heat'), 1e-9);
    }

    public function testEmptyByDefault(): void
    {
        $this->assertTrue((new Epigenome())->isEmpty());
        $this->assertSame(0.0, (new Epigenome())->intensity('stress'));
    }
}
