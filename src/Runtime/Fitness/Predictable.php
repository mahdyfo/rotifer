<?php

declare(strict_types=1);

namespace Rotifer\Runtime\Fitness;

use Rotifer\Organism\Organism;

/**
 * A problem that knows how to present its champion's results in a custom way -
 * e.g. a game reporting its score rather than a row-by-row input/output table.
 * Row-based problems don't need this; {@see Predictor} handles them generically.
 */
interface Predictable
{
    /**
     * @return array{columns: list<string>, rows: list<list<string>>}
     */
    public function describe(Organism $best): array;
}
