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
 */
final class Scorer
{
    public static function score(Organism $organism, Problem $problem): float
    {
        $organism->reset();
        $total = 0.0;
        foreach ($problem->data() as $row) {
            if ($row === []) {
                $organism->resetMemory();
                continue;
            }
            $organism->step($row[0]);
            $total += $problem->fitness($organism, $row);
        }
        $organism->resetMemory(); // clear state at the end of the cycle (legacy semantics)
        return $total;
    }

    public static function scoreGenome(Genome $genome, NetworkSpec $spec, Problem $problem): float
    {
        return self::score(new Organism($genome, $spec), $problem);
    }
}
