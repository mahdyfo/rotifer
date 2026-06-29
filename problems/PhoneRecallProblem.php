<?php

declare(strict_types=1);

namespace Rotifer\Problems;

use Rotifer\Network\Activation\Sigmoid;
use Rotifer\Network\Shape;
use Rotifer\Organism\Organism;
use Rotifer\Runtime\EvolutionConfig;
use Rotifer\Runtime\Fitness\Predictable;
use Rotifer\Runtime\Fitness\Problem;

/**
 * "Remember my phone number." Every step feeds the *same* constant tick, yet the
 * network must emit the next digit of a stored number - only possible with
 * internal memory, so it has to evolve a counter and read digits off it. A pure
 * test of recurrence. Digits are normalized to [0,1] (digit/9) for the sigmoid.
 */
final class PhoneRecallProblem implements Problem, Predictable
{
    /** The number to memorize, digit by digit. */
    private const PHONE = '01100110101';

    public function name(): string
    {
        return 'phone_recall';
    }

    public function shape(): Shape
    {
        return new Shape(inputs: 1, outputs: 1); // a single constant tick in, one digit out
    }

    public function config(): EvolutionConfig
    {
        return EvolutionConfig::default()
            ->name($this->name())
            ->population(120)
            ->generations(150)
            ->islands(3)
            ->surviveRate(0.3)
            ->elitism(0)
            ->initialHidden(3) // more room to build an internal counter
            ->memory(true) // the whole point: state must persist between ticks
            ->activation(new Sigmoid())
            ->crossover(0.5)
            ->mutation(weight: 0.4, addNeuron: 0.04, addConnection: 0.1, removeNeuron: 0.04, removeConnection: 0.1)
            ->weightMutation(count: 1, adjustmentRange: 0.8, randomizeProbability: 0.12)
            ->migration(30, 2)
            ->diversityInjection(0.06)
            ->adaptiveMutation(false)
            ->trauma(true)
            ->simplicity(5)
            ->parallel(true)
            ->seed(7000);
    }

    public function data(): array
    {
        $rows = [];
        foreach ($this->digits() as $digit) {
            $rows[] = [[1.0], [$digit]]; // identical input every step
        }
        return $rows;
    }

    public function fitness(Organism $organism, array $row): float
    {
        return 1.0 - abs($organism->outputs()[0] - $row[1][0]);
    }

    public function describe(Organism $best): array
    {
        $best->reset();
        $rows = [];
        $correct = 0;
        foreach ($this->digits() as $step => $digit) {
            $best->step([1.0]);
            $predicted = round($best->outputs()[0], 4);
            $ok = round($predicted) == $digit;
            $correct += $ok ? 1 : 0;
            $rows[] = [(string) $step, 1, (string) $digit, (string) $predicted, $ok ? 'ok' : '-'];
        }

        $total = count($rows);
        return [
            'columns' => ['step', 'input', 'expected', 'predicted', 'ok'],
            'rows' => $rows,
            'successRate' => $total > 0 ? $correct / $total : 0.0,
        ];
    }

    /** @return list<int> */
    private function digits(): array
    {
        return array_map(intval(...), str_split(self::PHONE));
    }
}
