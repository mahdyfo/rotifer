<?php

declare(strict_types=1);

namespace Rotifer\Tests\Unit\Network;

use PHPUnit\Framework\TestCase;
use Rotifer\Genome\Gene;
use Rotifer\Genome\Genome;
use Rotifer\Genome\NodeType;
use Rotifer\Network\Brain;
use Rotifer\Network\NetworkSpec;
use Rotifer\Network\Shape;
use Rotifer\Tests\Support\Identity;

final class BrainTest extends TestCase
{
    private function spec(int $inputs, int $outputs, bool $memory = false): NetworkSpec
    {
        return new NetworkSpec(new Shape($inputs, $outputs), $memory, new Identity());
    }

    /** @param list<array{NodeType,int,NodeType,int,float}> $tuples */
    private function genome(array $tuples): Genome
    {
        return new Genome(array_map(
            fn (array $t) => Gene::of($t[0], $t[1], $t[2], $t[3], $t[4]),
            $tuples,
        ));
    }

    public function testWeightedSumThroughDirectConnections(): void
    {
        $brain = new Brain($this->genome([
            [NodeType::Input, 0, NodeType::Output, 0, 2.0],
            [NodeType::Input, 1, NodeType::Output, 0, -1.0],
        ]), $this->spec(2, 1));

        $this->assertSame([5.0], $brain->step([3.0, 1.0])); // 3*2 + 1*-1
    }

    public function testFeedForwardThroughOrderedHiddenLayer(): void
    {
        $brain = new Brain($this->genome([
            [NodeType::Input, 0, NodeType::Hidden, 0, 1.0],
            [NodeType::Hidden, 0, NodeType::Hidden, 1, 3.0],
            [NodeType::Hidden, 1, NodeType::Output, 0, 1.0],
        ]), $this->spec(1, 1));

        // h0 = 2, h1 = 2*3 = 6, out = 6
        $this->assertSame([6.0], $brain->step([2.0]));
    }

    public function testOutputsAlwaysMatchOutputCountEvenWhenUnwired(): void
    {
        $brain = new Brain($this->genome([
            [NodeType::Input, 0, NodeType::Output, 1, 1.0],
        ]), $this->spec(1, 3));

        $this->assertSame([0.0, 5.0, 0.0], $brain->step([5.0]));
    }

    public function testMemoryAccumulatesAcrossSteps(): void
    {
        $brain = new Brain($this->genome([
            [NodeType::Input, 0, NodeType::Hidden, 0, 1.0],
            [NodeType::Hidden, 0, NodeType::Hidden, 0, 1.0], // self recurrence
            [NodeType::Hidden, 0, NodeType::Output, 0, 1.0],
        ]), $this->spec(1, 1, memory: true));

        $this->assertSame([1.0], $brain->step([1.0]));
        $this->assertSame([2.0], $brain->step([1.0]));
        $this->assertSame([3.0], $brain->step([1.0]));

        $brain->resetMemory();
        $this->assertSame([1.0], $brain->step([1.0]));
    }

    public function testActivationsExposeEveryNeuronValue(): void
    {
        $brain = new Brain($this->genome([
            [NodeType::Input, 0, NodeType::Hidden, 0, 2.0],
            [NodeType::Hidden, 0, NodeType::Output, 0, 1.0],
        ]), $this->spec(1, 1));

        $brain->step([3.0]);
        $activations = $brain->activations();

        $this->assertSame(3.0, $activations['0:0']); // input
        $this->assertSame(6.0, $activations['1:0']); // hidden = 3 * 2
        $this->assertSame(6.0, $activations['2:0']); // output
    }

    public function testWithoutMemoryEachStepIsIndependentAndSelfLoopPruned(): void
    {
        $brain = new Brain($this->genome([
            [NodeType::Input, 0, NodeType::Hidden, 0, 1.0],
            [NodeType::Hidden, 0, NodeType::Hidden, 0, 1.0], // pruned (no memory)
            [NodeType::Hidden, 0, NodeType::Output, 0, 1.0],
        ]), $this->spec(1, 1, memory: false));

        $this->assertSame(1, $brain->hiddenCount());
        $this->assertFalse($brain->genome()->has('1:0->1:0'));
        $this->assertSame([1.0], $brain->step([1.0]));
        $this->assertSame([1.0], $brain->step([1.0])); // no accumulation
    }
}
