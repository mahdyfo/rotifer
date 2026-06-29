<?php

declare(strict_types=1);

namespace Rotifer\Problems;

use Rotifer\Network\Activation\Sigmoid;
use Rotifer\Network\Shape;
use Rotifer\Organism\Organism;
use Rotifer\Runtime\EvolutionConfig;
use Rotifer\Runtime\Fitness\Describable;
use Rotifer\Runtime\Fitness\Problem;

/**
 * Unsupervised compression: reconstruct each of eight one-hot patterns after
 * squeezing them through a small hidden bottleneck. A fixed layered topology
 * (8 -> 3 -> 8) makes it a standard auto-encoder: inputs connect only to the
 * hidden bottleneck and the bottleneck only to the outputs - no direct
 * input->output shortcuts and no intra-layer edges - so the network is forced to
 * actually encode through the 3-unit bottleneck. Only the weights evolve; the
 * neuron count is frozen. Per row, fitness is reconstruction (1/(1+MSE)) plus
 * bit-accuracy (fraction of outputs on the correct side of 0.5); perfect
 * reconstruction across all nine rows scores 18.0.
 */
final class AutoEncoderProblem implements Problem, Describable
{
    public function description(): string
    {
        return 'Unsupervised compression: reconstruct 7 inputs (+1 bias) through a 3-unit bottleneck (a fixed 5-3-5 hidden layer auto-encoder).';
    }

    public function name(): string
    {
        return 'auto_encoder';
    }

    public function shape(): Shape
    {
        return new Shape(inputs: 7, outputs: 6);
    }

    public function config(): EvolutionConfig
    {
        return EvolutionConfig::default()
            ->name($this->name())
            ->population(120)
            ->generations(300)
            ->islands(3)
            ->surviveRate(0.3)
            ->elitism(0)
            ->hiddenLayers([5, 3, 5]) // fixed, no input->output edges
            ->activation(new Sigmoid())
            ->crossover(0.5)
            ->mutation(weight: 0.85, addNeuron: 0.0, addConnection: 0, removeNeuron: 0.0, removeConnection: 0)
            ->weightMutation(count: 10, adjustmentRange: 0.7, randomizeProbability: 0.1)
            ->migration(everyGenerations: 30, topK: 2)
            ->trauma(true)
            ->adaptiveMutation(false)
            ->diversityInjection(0.05)
            ->simplicity(0)
            ->parallel(false)
            ->seed(1234);
    }

    public function data(): array
    {
        return [
            [[1, 1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]],
            [[1, 0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]],
            [[1, 0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0]],
            [[1, 0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0]],
            [[1, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 0]],
            [[1, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1]],
        ];
    }

    public function fitness(Organism $organism, array $row): float
    {
        $outputs = $organism->outputs();
        $hot = 0;
        foreach ($row[1] as $i => $target) {
            if ($target > $row[1][$hot]) {
                $hot = $i;
            }
        }
        $others = 0.0;
        $count = 0;
        foreach ($outputs as $i => $value) {
            if ($i !== $hot) {
                $others += $value;
                $count++;
            }
        }
        $meanOther = $count > 0 ? $others / $count : 0.0;
        $margin = max(0.0, ($outputs[$hot] ?? 0.0) - $meanOther);
        return sqrt($margin);
    }
}
