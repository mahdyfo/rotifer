<?php

declare(strict_types=1);

namespace Rotifer\Tests\Unit\Network;

use PHPUnit\Framework\TestCase;
use Rotifer\Genome\Gene;
use Rotifer\Genome\Genome;
use Rotifer\Genome\NodeType;
use Rotifer\Network\GenomePruner;
use Rotifer\Network\LayerPlan;

final class GenomePrunerTest extends TestCase
{
    /** @param list<array{NodeType,int,NodeType,int,float}> $tuples */
    private function genome(array $tuples): Genome
    {
        return new Genome(array_map(
            fn (array $t) => Gene::of($t[0], $t[1], $t[2], $t[3], $t[4]),
            $tuples,
        ));
    }

    /** @return list<string> */
    private function keys(Genome $g): array
    {
        return array_map(fn (Gene $gene) => $gene->connectionKey(), $g->genes());
    }

    public function testDropsEdgesIntoInputsAndOutOfOutputs(): void
    {
        $pruned = GenomePruner::prune($this->genome([
            [NodeType::Input, 0, NodeType::Output, 0, 1.0],   // legal
            [NodeType::Hidden, 0, NodeType::Input, 0, 1.0],   // into input -> illegal
            [NodeType::Output, 0, NodeType::Hidden, 0, 1.0],  // out of output -> illegal
        ]), hasMemory: true);

        $this->assertSame(['0:0->2:0'], $this->keys($pruned));
    }

    public function testWithoutMemoryDropsSelfAndBackwardHiddenEdges(): void
    {
        $pruned = GenomePruner::prune($this->genome([
            [NodeType::Input, 0, NodeType::Hidden, 0, 1.0],
            [NodeType::Input, 0, NodeType::Hidden, 1, 1.0],
            [NodeType::Hidden, 0, NodeType::Hidden, 0, 1.0],  // self -> dropped
            [NodeType::Hidden, 1, NodeType::Hidden, 0, 1.0],  // backward (1 -> 0) -> dropped
            [NodeType::Hidden, 0, NodeType::Hidden, 1, 1.0],  // forward (0 -> 1) -> kept
            [NodeType::Hidden, 1, NodeType::Output, 0, 1.0],
        ]), hasMemory: false);

        $keys = $this->keys($pruned);
        $this->assertContains('1:0->1:1', $keys);
        $this->assertNotContains('1:0->1:0', $keys);
        $this->assertNotContains('1:1->1:0', $keys);
    }

    public function testWithMemoryKeepsSelfAndRecurrentEdges(): void
    {
        $pruned = GenomePruner::prune($this->genome([
            [NodeType::Input, 0, NodeType::Hidden, 0, 1.0],
            [NodeType::Hidden, 0, NodeType::Hidden, 0, 1.0],  // self loop kept under memory
            [NodeType::Hidden, 0, NodeType::Output, 0, 1.0],
        ]), hasMemory: true);

        $this->assertContains('1:0->1:0', $this->keys($pruned));
    }

    public function testPrunesHiddenWithoutAnOutput(): void
    {
        $pruned = GenomePruner::prune($this->genome([
            [NodeType::Input, 0, NodeType::Output, 0, 1.0],
            [NodeType::Input, 0, NodeType::Hidden, 0, 1.0], // h0 has input but no output
        ]), hasMemory: false);

        $this->assertSame(['0:0->2:0'], $this->keys($pruned));
    }

    public function testCollapsesDeadChainsToFixpoint(): void
    {
        // input -> h0 -> h1, but h1 reaches no output: the whole chain dies.
        $pruned = GenomePruner::prune($this->genome([
            [NodeType::Input, 0, NodeType::Hidden, 0, 1.0],
            [NodeType::Hidden, 0, NodeType::Hidden, 1, 1.0],
        ]), hasMemory: false);

        $this->assertSame([], $this->keys($pruned));
        $this->assertTrue($pruned->isEmpty());
    }

    public function testKeepsAValidInputToOutputChain(): void
    {
        $pruned = GenomePruner::prune($this->genome([
            [NodeType::Input, 0, NodeType::Hidden, 0, 1.0],
            [NodeType::Hidden, 0, NodeType::Hidden, 1, 1.0],
            [NodeType::Hidden, 1, NodeType::Output, 0, 1.0],
        ]), hasMemory: false);

        $this->assertCount(3, $pruned->genes());
    }

    public function testLayeredPlanDropsShortcutsAndSkips(): void
    {
        // Two hidden layers: layer0 = {0,1}, layer1 = {2}. Only consecutive-layer
        // edges survive: input->layer0, layer0->layer1, layer1->output.
        $pruned = GenomePruner::prune($this->genome([
            [NodeType::Input, 0, NodeType::Hidden, 0, 1.0],   // input -> layer0  (kept)
            [NodeType::Hidden, 0, NodeType::Hidden, 2, 1.0],  // layer0 -> layer1 (kept)
            [NodeType::Hidden, 2, NodeType::Output, 0, 1.0],  // layer1 -> output (kept)
            [NodeType::Input, 0, NodeType::Output, 0, 1.0],   // input -> output  (shortcut, dropped)
            [NodeType::Input, 0, NodeType::Hidden, 2, 1.0],   // input -> layer1  (skip, dropped)
            [NodeType::Hidden, 0, NodeType::Hidden, 1, 1.0],  // within layer0    (intra-layer, dropped)
            [NodeType::Hidden, 0, NodeType::Output, 0, 1.0],  // layer0 -> output (skip, dropped)
        ]), hasMemory: false, layers: new LayerPlan([2, 1]));

        $this->assertSame(['0:0->1:0', '1:0->1:2', '1:2->2:0'], $this->keys($pruned));
    }

    public function testLayeredPlanKeepsNeuronMissingAnOutputEdge(): void
    {
        // Unlike dynamic mode, a fixed layered network never prunes a planned
        // neuron: hidden 0 keeps its input edge even with no outgoing edge.
        $pruned = GenomePruner::prune($this->genome([
            [NodeType::Input, 0, NodeType::Hidden, 0, 1.0],
        ]), hasMemory: false, layers: new LayerPlan([1]));

        $this->assertSame(['0:0->1:0'], $this->keys($pruned));
    }
}
