<?php

declare(strict_types=1);

namespace Rotifer\Tests\Unit\Network;

use PHPUnit\Framework\TestCase;
use Rotifer\Genome\NodeType;
use Rotifer\Network\LayerPlan;

final class LayerPlanTest extends TestCase
{
    public function testNumbersHiddenNeuronsContiguouslyPerLayer(): void
    {
        $plan = new LayerPlan([3, 2]);

        $this->assertSame(5, $plan->totalHidden());
        $this->assertSame(2, $plan->layerCount());
        $this->assertSame([0, 1, 2], $plan->hiddenInLayer(0));
        $this->assertSame([3, 4], $plan->hiddenInLayer(1));
        $this->assertSame(0, $plan->layerOfHidden(2));
        $this->assertSame(1, $plan->layerOfHidden(3));
        $this->assertSame(-1, $plan->layerOfHidden(5));
    }

    public function testAllowsOnlyOneLayerForwardEdges(): void
    {
        $plan = new LayerPlan([2, 1]); // input -> {0,1} -> {2} -> output

        $this->assertTrue($plan->allows(NodeType::Input, 0, NodeType::Hidden, 0));
        $this->assertTrue($plan->allows(NodeType::Hidden, 1, NodeType::Hidden, 2));
        $this->assertTrue($plan->allows(NodeType::Hidden, 2, NodeType::Output, 0));

        // Shortcut, skip, intra-layer and backward edges are all rejected.
        $this->assertFalse($plan->allows(NodeType::Input, 0, NodeType::Output, 0));
        $this->assertFalse($plan->allows(NodeType::Input, 0, NodeType::Hidden, 2));
        $this->assertFalse($plan->allows(NodeType::Hidden, 0, NodeType::Hidden, 1));
        $this->assertFalse($plan->allows(NodeType::Hidden, 2, NodeType::Hidden, 0));
        $this->assertFalse($plan->allows(NodeType::Hidden, 0, NodeType::Output, 0));
    }

    public function testWithNoHiddenLayersInputConnectsStraightToOutput(): void
    {
        $plan = new LayerPlan([]);

        $this->assertSame(0, $plan->layerCount());
        $this->assertTrue($plan->allows(NodeType::Input, 0, NodeType::Output, 0));
    }
}
