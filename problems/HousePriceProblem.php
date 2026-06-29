<?php

declare(strict_types=1);

namespace Rotifer\Problems;

use Rotifer\Network\Activation\Sigmoid;
use Rotifer\Network\Shape;
use Rotifer\Organism\Organism;
use Rotifer\Runtime\EvolutionConfig;
use Rotifer\Runtime\Fitness\Problem;

/**
 * A regression task: estimate a normalized house price from a handful of
 * normalized features (bias, bedrooms, bathrooms, area, age, distance-to-city).
 * Shows how to point Rotifer at ordinary tabular data. Perfect fit approaches
 * one point per row.
 */
final class HousePriceProblem implements Problem
{
    public function name(): string
    {
        return 'house_price';
    }

    public function shape(): Shape
    {
        return new Shape(inputs: 6, outputs: 1); // bias + five features
    }

    public function config(): EvolutionConfig
    {
        return EvolutionConfig::default()
            ->name($this->name())
            ->population(150)
            ->generations(500)
            ->islands(2)
            ->surviveRate(0.2)
            ->elitism(0)
            ->initialHidden(3)
            ->activation(new Sigmoid())
            ->crossover(0.5)
            ->mutation(weight: 0.4, addNeuron: 0.08, addConnection: 0.1, removeNeuron: 0.08, removeConnection: 0.1)
            ->weightMutation(count: 2, adjustmentRange: 0.6, randomizeProbability: 0.1)
            ->migration(20, 2)
            ->adaptiveMutation(true)
            ->diversityInjection(0.1)
            ->simplicity(0)
            ->parallel(true)
            ->seed(10);
    }

    public function data(): array
    {
        // [bias, bedrooms, bathrooms, area, age, distance] -> [price]  (all 0..1)
        return [
            [[1, 0.2, 0.2, 0.15, 0.8, 0.9], [0.15]],
            [[1, 0.4, 0.4, 0.40, 0.5, 0.6], [0.40]],
            [[1, 0.6, 0.6, 0.65, 0.3, 0.4], [0.62]],
            [[1, 0.8, 0.8, 0.85, 0.1, 0.2], [0.85]],
            [[1, 1.0, 0.8, 1.00, 0.0, 0.1], [0.95]],
            [[1, 0.3, 0.4, 0.30, 0.6, 0.7], [0.30]],
            [[1, 0.7, 0.6, 0.70, 0.2, 0.3], [0.70]],
            [[1, 0.5, 0.4, 0.50, 0.4, 0.5], [0.48]],
        ];
    }

    public function fitness(Organism $organism, array $row): float
    {
        return 1.0 - abs($organism->outputs()[0] - $row[1][0]);
    }
}
