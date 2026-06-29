<?php

declare(strict_types=1);

namespace Rotifer\Runtime\Fitness;

use Rotifer\Organism\Organism;

/**
 * Produces the "what did the champion actually predict" table shown at the end
 * of a run (in the terminal and the web dashboard).
 *
 * For an ordinary row-based problem it runs the best organism over the data,
 * lines up expected vs predicted, and scores each row by how close the two are
 * (plus an overall success rate). A problem that implements {@see Predictable}
 * (e.g. a game) supplies its own table instead.
 *
 * @return array{columns: list<string>, rows: list<list<string>>, successRate?: float}
 */
final class Predictor
{
    public static function describe(Problem $problem, Organism $best): array
    {
        if ($problem instanceof Predictable) {
            return $problem->describe($best);
        }

        $best->reset();
        $rows = [];
        $index = 0;
        $matchSum = 0.0;
        $scored = 0;
        foreach ($problem->data() as $row) {
            if ($row === []) {
                $best->resetMemory();
                continue;
            }
            $best->step($row[0]);
            $match = self::closeness($best->outputs(), $row[1]);
            $matchSum += $match;
            $scored++;
            $rows[] = [
                (string) $index++,
                self::vector($row[0]),
                self::vector($row[1]),
                self::vector($best->outputs()),
                self::percent($match),
            ];
        }

        return [
            'columns' => ['#', 'input', 'expected', 'predicted', 'match'],
            'rows' => $rows,
            'successRate' => $scored > 0 ? $matchSum / $scored : 0.0,
        ];
    }

    /**
     * The champion's overall match rate in [0,1], or null when it has no
     * meaningful one (e.g. an episodic game). Cheap enough to show live.
     */
    public static function successRate(Problem $problem, Organism $organism): ?float
    {
        if ($problem instanceof Predictable) {
            return $problem->describe($organism)['successRate'] ?? null;
        }

        $organism->reset();
        $sum = 0.0;
        $n = 0;
        foreach ($problem->data() as $row) {
            if ($row === []) {
                $organism->resetMemory();
                continue;
            }
            $organism->step($row[0]);
            $sum += self::closeness($organism->outputs(), $row[1]);
            $n++;
        }
        return $n > 0 ? $sum / $n : null;
    }

    /**
     * Closeness in [0,1] (1 = exact): 1 minus mean absolute error per output.
     *
     * @param list<float> $predicted
     * @param list<float> $expected
     */
    private static function closeness(array $predicted, array $expected): float
    {
        if ($expected === []) {
            return 1.0;
        }
        $error = 0.0;
        foreach ($expected as $i => $target) {
            $error += abs((float) $target - (float) ($predicted[$i] ?? 0.0));
        }
        return max(0.0, 1.0 - $error / count($expected));
    }

    private static function percent(float $fraction): string
    {
        return round($fraction * 100) . '%';
    }

    /** @param list<float> $values */
    private static function vector(array $values): string
    {
        return '[' . implode(', ', array_map(static fn (float $v) => self::number($v), $values)) . ']';
    }

    private static function number(float $v): string
    {
        $rounded = round($v, 3);
        return $rounded == (int) $rounded ? (string) (int) $rounded : (string) $rounded;
    }
}
