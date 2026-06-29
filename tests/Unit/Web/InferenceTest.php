<?php

declare(strict_types=1);

namespace Rotifer\Tests\Unit\Web;

use PHPUnit\Framework\TestCase;
use Rotifer\Web\Inference;

final class InferenceTest extends TestCase
{
    public function testRunsSavedGenesAndReportsEveryNeuron(): void
    {
        // input0 --2--> hidden0 --1--> output0, identity-ish via relu (positive stays).
        $genes = [
            [0, 0, 1, 0, 2.0],
            [1, 0, 2, 0, 1.0],
        ];

        $result = Inference::evaluate($genes, inputs: 1, outputs: 1, activation: 'relu', memory: false, steps: [[3.0]]);

        $this->assertSame([6.0], $result['outputs']);     // relu(relu(3*2)*1)
        $this->assertSame(3.0, $result['nodes']['0:0']);  // input
        $this->assertSame(6.0, $result['nodes']['1:0']);  // hidden
        $this->assertSame(6.0, $result['nodes']['2:0']);  // output
    }

    public function testMissingInputsAreZeroFilled(): void
    {
        $genes = [[0, 1, 2, 0, 1.0]]; // reads input index 1
        $result = Inference::evaluate($genes, inputs: 2, outputs: 1, activation: 'relu', memory: false, steps: [[5.0]]);

        $this->assertSame([0.0], $result['outputs']); // input 1 defaulted to 0
    }

    public function testMemoryAccumulatesAcrossSteps(): void
    {
        // input0 --1--> hidden0 --1--> hidden0 (self) --1--> output0, memory on.
        $genes = [
            [0, 0, 1, 0, 1.0],
            [1, 0, 1, 0, 1.0],
            [1, 0, 2, 0, 1.0],
        ];
        $result = Inference::evaluate($genes, inputs: 1, outputs: 1, activation: 'relu', memory: true, steps: [[1.0], [1.0], [1.0]]);

        $this->assertSame([3.0], $result['outputs']); // accumulates 1, 2, 3 over the steps
    }
}
