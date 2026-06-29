<?php

declare(strict_types=1);

namespace Rotifer\Tests\Unit\Genome;

use PHPUnit\Framework\TestCase;
use Rotifer\Genome\Gene;
use Rotifer\Genome\Genome;
use Rotifer\Genome\NodeType;

final class GenomeTest extends TestCase
{
    private function gene(int $from, int $to, float $w): Gene
    {
        return Gene::of(NodeType::Input, $from, NodeType::Hidden, $to, $w);
    }

    public function testDeduplicatesByConnectionKeepingTheLastWeight(): void
    {
        $genome = new Genome([
            $this->gene(0, 0, 1.0),
            $this->gene(0, 0, 2.0),
        ]);

        $this->assertSame(1, $genome->count());
        $this->assertSame(2.0, $genome->genes()[0]->weight);
    }

    public function testWithAndWithoutAreImmutable(): void
    {
        $base = new Genome([$this->gene(0, 0, 1.0)]);
        $added = $base->with($this->gene(1, 0, 0.5));
        $removed = $added->without('0:0->1:0');

        $this->assertSame(1, $base->count(), 'original untouched');
        $this->assertSame(2, $added->count());
        $this->assertSame(1, $removed->count());
        $this->assertFalse($removed->has('0:0->1:0'));
        $this->assertTrue($removed->has('0:1->1:0'));
    }

    public function testDistanceToSelfIsZero(): void
    {
        $genome = new Genome([$this->gene(0, 0, 1.0), $this->gene(1, 0, 2.0)]);
        $this->assertSame(0.0, $genome->distanceTo($genome));
    }

    public function testDistanceCountsDisjointGenes(): void
    {
        $a = new Genome([$this->gene(0, 0, 1.0), $this->gene(1, 0, 1.0)]);
        $b = new Genome([$this->gene(0, 0, 1.0)]); // missing one of a's two genes

        // 1 disjoint / larger size 2 = 0.5, weight diff 0 on the shared gene
        $this->assertSame(0.5, $a->distanceTo($b, 1.0, 0.4));
    }

    public function testDistanceCountsWeightDifference(): void
    {
        $a = new Genome([$this->gene(0, 0, 1.0)]);
        $b = new Genome([$this->gene(0, 0, 3.0)]);

        // no disjoint genes; mean weight diff 2.0 * coefficient 0.5 = 1.0
        $this->assertSame(1.0, $a->distanceTo($b, 1.0, 0.5));
    }

    public function testArrayRoundTripPreservesOrder(): void
    {
        $genome = new Genome([$this->gene(0, 0, 1.0), $this->gene(2, 1, -0.5)]);
        $restored = Genome::fromArray($genome->toArray());

        $this->assertSame($genome->count(), $restored->count());
        $this->assertSame(
            array_map(fn (Gene $g) => $g->connectionKey(), $genome->genes()),
            array_map(fn (Gene $g) => $g->connectionKey(), $restored->genes()),
        );
    }
}
