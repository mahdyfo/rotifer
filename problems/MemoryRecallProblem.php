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
 * A temporal task: at each step output the signal seen on the *previous* step
 * (a one-step delay line). Impossible without internal state, so it only works
 * for memory-enabled networks - the showcase for recurrence. Perfect score
 * equals the number of scored steps.
 */
final class MemoryRecallProblem implements Problem, Describable
{
    public function description(): string
    {
        return 'Output the signal seen on the previous step - a one-step delay line that needs evolved recurrence.';
    }

    /** The driving signal sequence (read alongside a constant bias). */
    private const SIGNAL = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0];

    public function name(): string
    {
        return 'memory_recall';
    }

    public function shape(): Shape
    {
        return new Shape(inputs: 2, outputs: 1); // bias + signal
    }

    public function config(): EvolutionConfig
    {
        return EvolutionConfig::default()
            ->name($this->name())
            ->population(100)
            ->generations(250)
            ->surviveRate(0.3)
            ->elitism(0)
            ->islands(2)
            ->initialHidden(0)
            ->memory(true)
            ->activation(new Sigmoid())
            ->crossover(0.5)
            ->mutation(weight: 0.85, addNeuron: 0.08, addConnection: 0.1, removeNeuron: 0.08, removeConnection: 0.2)
            ->weightMutation(count: 1, adjustmentRange: 0.8, randomizeProbability: 0.12)
            ->diversityInjection(0.05)
            ->migration(20, 2)
            ->simplicity(1)
            ->parallel(true)
            ->seed(1000);
    }

    public function data(): array
    {
        $rows = [];
        $previous = 0;
        foreach (self::SIGNAL as $signal) {
            $rows[] = [[1, $signal], [(float) $previous]];
            $previous = $signal;
        }
        return $rows;
    }

    public function fitness(Organism $organism, array $row): float
    {
        return 1.0 - abs($organism->outputs()[0] - $row[1][0]);
    }
}
