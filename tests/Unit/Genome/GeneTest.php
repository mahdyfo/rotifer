<?php

declare(strict_types=1);

namespace Rotifer\Tests\Unit\Genome;

use PHPUnit\Framework\TestCase;
use Rotifer\Genome\Gene;
use Rotifer\Genome\NodeType;
use Rotifer\Genome\Weight;

final class GeneTest extends TestCase
{
    public function testConnectionKeyIdentifiesTheEdgeIgnoringWeight(): void
    {
        $a = Gene::of(NodeType::Input, 0, NodeType::Hidden, 2, 1.5);
        $b = Gene::of(NodeType::Input, 0, NodeType::Hidden, 2, -3.0);
        $this->assertSame('0:0->1:2', $a->connectionKey());
        $this->assertSame($a->connectionKey(), $b->connectionKey());
    }

    public function testWeightIsClampedToTheLegalRange(): void
    {
        $gene = Gene::of(NodeType::Input, 0, NodeType::Output, 0, 999.0);
        $this->assertSame(Weight::MAX, $gene->weight);

        $negative = Gene::of(NodeType::Input, 0, NodeType::Output, 0, -999.0);
        $this->assertSame(-Weight::MAX, $negative->weight);
    }

    public function testWithWeightReturnsANewGeneOnTheSameConnection(): void
    {
        $gene = Gene::of(NodeType::Hidden, 1, NodeType::Output, 0, 0.5);
        $changed = $gene->withWeight(1.25);

        $this->assertSame(0.5, $gene->weight, 'original is untouched');
        $this->assertSame(1.25, $changed->weight);
        $this->assertSame($gene->connectionKey(), $changed->connectionKey());
    }

    public function testArrayRoundTrip(): void
    {
        $gene = Gene::of(NodeType::Hidden, 4, NodeType::Output, 1, -2.345678);
        $restored = Gene::fromArray($gene->toArray());

        $this->assertTrue($gene->from->equals($restored->from));
        $this->assertTrue($gene->to->equals($restored->to));
        $this->assertSame($gene->weight, $restored->weight);
    }
}
