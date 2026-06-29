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
 * Multi-class classification: map weather readings (bias, temperature, humidity,
 * pressure, wind) to one of four conditions via four output neurons, picking the
 * highest. Fitness rewards a correct argmax plus low error, so a perfect
 * classifier approaches two points per row.
 */
final class WeatherForecastProblem implements Problem, Describable
{
    public function description(): string
    {
        return 'Multi-class classification: map weather readings to one of four conditions via an argmax over four outputs.';
    }

    /** sunny, cloudy, rainy, stormy */
    private const CLASSES = 4;

    public function name(): string
    {
        return 'weather_forecast';
    }

    public function shape(): Shape
    {
        return new Shape(inputs: 5, outputs: self::CLASSES);
    }

    public function config(): EvolutionConfig
    {
        return EvolutionConfig::default()
            ->name($this->name())
            ->population(200)
            ->generations(500)
            ->islands(3)
            ->surviveRate(0.45)
            ->elitism(0)
            ->initialHidden(4)
            ->activation(new Sigmoid())
            ->crossover(0.5)
            ->mutation(weight: 0.4, addNeuron: 0.04, addConnection: 0.1, removeNeuron: 0.04, removeConnection: 0.1)
            ->weightMutation(count: 2, adjustmentRange: 0.7, randomizeProbability: 0.1)
            ->adaptiveMutation(true)
            ->trauma(false)
            ->migration(everyGenerations: 50, topK: 2)
            ->diversityInjection(0.1)
            ->parallel(true)
            ->simplicity(2)
            ->seed(1000);
    }

    public function data(): array
    {
        // [bias, temp, humidity, pressure, wind] -> one-hot [sunny, cloudy, rainy, stormy]
        return [
            [[1, 0.9, 0.2, 0.8, 0.2], [1, 0, 0, 0]],
            [[1, 0.8, 0.3, 0.7, 0.1], [1, 0, 0, 0]],
            [[1, 0.7, 0.4, 0.6, 0.3], [1, 0, 0, 0]],
            [[1, 0.5, 0.6, 0.5, 0.4], [0, 1, 0, 0]],
            [[1, 0.6, 0.5, 0.5, 0.3], [0, 1, 0, 0]],
            [[1, 0.4, 0.6, 0.4, 0.5], [0, 1, 0, 0]],
            [[1, 0.4, 0.8, 0.4, 0.5], [0, 0, 1, 0]],
            [[1, 0.3, 0.9, 0.3, 0.6], [0, 0, 1, 0]],
            [[1, 0.5, 0.8, 0.3, 0.5], [0, 0, 1, 0]],
            [[1, 0.3, 0.9, 0.2, 0.9], [0, 0, 0, 1]],
            [[1, 0.2, 1.0, 0.1, 1.0], [0, 0, 0, 1]],
            [[1, 0.4, 0.9, 0.2, 0.8], [0, 0, 0, 1]],
        ];
    }

    public function fitness(Organism $organism, array $row): float
    {
        $outputs = $organism->outputs();
        $target = $row[1];

        $error = 0.0;
        foreach ($target as $i => $t) {
            $error += abs($t - ($outputs[$i] ?? 0.0));
        }
        $accuracy = 1.0 - $error / self::CLASSES;

        $correct = $this->argmax($outputs) === $this->argmax($target) ? 1.0 : 0.0;
        return $accuracy + $correct;
    }

    /** @param list<float> $values */
    private function argmax(array $values): int
    {
        $best = 0;
        foreach ($values as $i => $v) {
            if ($v > $values[$best]) {
                $best = $i;
            }
        }
        return $best;
    }
}
