<?php

declare(strict_types=1);

namespace Rotifer\Problems;

use Rotifer\Network\Activation\Sigmoid;
use Rotifer\Network\Shape;
use Rotifer\Organism\Organism;
use Rotifer\Runtime\EvolutionConfig;
use Rotifer\Runtime\Fitness\Problem;

/**
 * The classic non-linearly-separable XOR. The smallest showcase of evolving
 * architecture: starting from a couple of hidden neurons, structural mutation is
 * free to grow whatever topology solves it. Perfect score is 4.0 (one per row).
 */
final class XorProblem implements Problem
{
    public function name(): string
    {
        return 'xor';
    }

    public function shape(): Shape
    {
        return new Shape(inputs: 3, outputs: 1); // bias + two operands
    }

    public function config(): EvolutionConfig
    {
        return EvolutionConfig::default()
            ->name($this->name())
            ->population(20)
            ->generations(30)
            ->islands(2)
            ->surviveRate(0.3)
            ->elitism(0)
            ->initialHidden(2)
            ->activation(new Sigmoid())
            ->crossover(0.5)
            ->mutation(weight: 0.4, addNeuron: 0.05, addConnection: 0.1, removeNeuron: 0.1, removeConnection: 0.1)
            ->weightMutation(count: 2, adjustmentRange: 0.8, randomizeProbability: 0.1)
            ->diversityInjection(0.05)
            ->adaptiveMutation(false)
            ->trauma(false)
            ->migration(everyGenerations: 8, topK: 2)
            ->seed(1234);
    }

    public function data(): array
    {
        return [
            [[1, 0, 0], [0]],
            [[1, 0, 1], [1]],
            [[1, 1, 0], [1]],
            [[1, 1, 1], [0]],
        ];
    }

    public function fitness(Organism $organism, array $row): float
    {
        return 1.0 - abs($organism->outputs()[0] - $row[1][0]);
    }
}
