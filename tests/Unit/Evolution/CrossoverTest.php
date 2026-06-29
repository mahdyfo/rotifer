<?php

declare(strict_types=1);

namespace Rotifer\Tests\Unit\Evolution;

use PHPUnit\Framework\TestCase;
use Rotifer\Evolution\Reproduction\Crossover;
use Rotifer\Genome\Gene;
use Rotifer\Genome\Genome;
use Rotifer\Genome\NodeType;
use Rotifer\Runtime\Rng;

final class CrossoverTest extends TestCase
{
    private function gene(int $from, int $to, float $w): Gene
    {
        return Gene::of(NodeType::Input, $from, NodeType::Output, $to, $w);
    }

    public function testSharedConnectionTakesFirstParentWeightWhenBiasIsZero(): void
    {
        $a = new Genome([$this->gene(0, 0, 1.0)]);
        $b = new Genome([$this->gene(0, 0, 8.0)]);

        $child = (new Crossover())->cross($a, 1.0, $b, 1.0, 0.0, new Rng(1));
        $this->assertSame(1.0, $child->get('0:0->2:0')->weight);
    }

    public function testSharedConnectionTakesSecondParentWeightWhenBiasIsOne(): void
    {
        $a = new Genome([$this->gene(0, 0, 1.0)]);
        $b = new Genome([$this->gene(0, 0, 8.0)]);

        $child = (new Crossover())->cross($a, 1.0, $b, 1.0, 1.0, new Rng(1));
        $this->assertSame(8.0, $child->get('0:0->2:0')->weight);
    }

    public function testDisjointGenesComeFromTheFitterParent(): void
    {
        $fit = new Genome([$this->gene(0, 0, 1.0), $this->gene(1, 0, 2.0)]);
        $weak = new Genome([$this->gene(0, 0, 1.0)]);

        $child = (new Crossover())->cross($weak, 0.0, $fit, 5.0, 0.0, new Rng(1));
        $this->assertTrue($child->has('0:1->2:0'), 'disjoint gene of the fitter parent is inherited');

        $childReversed = (new Crossover())->cross($fit, 5.0, $weak, 0.0, 0.0, new Rng(1));
        $this->assertTrue($childReversed->has('0:1->2:0'));
    }

    public function testEqualFitnessUnionsDisjointGenes(): void
    {
        $a = new Genome([$this->gene(0, 0, 1.0)]);
        $b = new Genome([$this->gene(1, 0, 2.0)]);

        $child = (new Crossover())->cross($a, 3.0, $b, 3.0, 0.0, new Rng(1));
        $this->assertSame(2, $child->count());
    }
}
