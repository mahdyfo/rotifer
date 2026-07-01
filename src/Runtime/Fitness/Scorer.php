<?php

declare(strict_types=1);

namespace Rotifer\Runtime\Fitness;

use Rotifer\Genome\Genome;
use Rotifer\Network\NetworkSpec;
use Rotifer\Organism\Organism;

/**
 * The one place that knows how to turn "an organism + a problem's data" into a
 * fitness number: reset, walk the rows (empty row = memory reset), sum the
 * problem's per-row score. Both the evaluator and the lifetime learner score
 * through here, so the rule lives in exactly one spot.
 *
 * An optional {@see ScoringWindow} narrows scoring to a randomly-placed slice of
 * the scorable rows (the anti-step-counting mechanism): rows before `start-prime`
 * are skipped (a cold start), the `prime` rows just before the window are fed
 * unscored to build memory context, rows in the window are scored, and rows past
 * it are dropped. Without a window every row is scored - the original behaviour.
 */
final class Scorer
{
    public static function score(Organism $organism, Problem $problem, ?ScoringWindow $window = null): float
    {
        $organism->reset();
        $total = 0.0;
        $index = 0; // position among scorable (non-empty) rows
        foreach ($problem->data() as $row) {
            if ($row === []) {
                $organism->resetMemory();
                continue;
            }
            if ($window !== null) {
                if ($index < $window->primeStart()) {
                    $index++; // cold start: not fed at all
                    continue;
                }
                if ($index < $window->start) {
                    $organism->step($row[0]); // priming rows: fed, unscored
                    $index++;
                    continue;
                }
                if ($index >= $window->end()) {
                    break; // past the window: nothing left to score
                }
            }
            $organism->step($row[0]);
            $total += $problem->fitness($organism, $row);
            $index++;
        }
        $organism->resetMemory(); // clear state at the end of the cycle (legacy semantics)
        return $total;
    }

    public static function scoreGenome(Genome $genome, NetworkSpec $spec, Problem $problem, ?ScoringWindow $window = null): float
    {
        return self::score(new Organism($genome, $spec), $problem, $window);
    }
}
