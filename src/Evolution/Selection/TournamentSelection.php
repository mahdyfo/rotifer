<?php

declare(strict_types=1);

namespace Rotifer\Evolution\Selection;

use Rotifer\Organism\Organism;
use Rotifer\Runtime\Rng;

/**
 * Tournament selection: sample k candidates at random and return the fittest.
 *
 * Small tournaments keep selection pressure moderate, which preserves diversity
 * better than always breeding the global best - the same reasoning the legacy
 * engine used, but expressed as a plain, testable pick.
 */
final class TournamentSelection
{
    public function __construct(private readonly int $tournamentSize = 3)
    {
    }

    /**
     * @param list<Organism> $candidates
     * @param (callable(Organism): float)|null $fitnessOf ranking metric; defaults
     *        to raw fitness.
     */
    public function pick(array $candidates, Rng $rng, ?callable $fitnessOf = null): Organism
    {
        $fitnessOf ??= static fn (Organism $o): float => $o->fitness();

        $best = $candidates[$rng->intBetween(0, count($candidates) - 1)];
        $bestScore = $fitnessOf($best);
        for ($i = 1; $i < $this->tournamentSize; $i++) {
            $contender = $candidates[$rng->intBetween(0, count($candidates) - 1)];
            $score = $fitnessOf($contender);
            if ($score > $bestScore) {
                $best = $contender;
                $bestScore = $score;
            }
        }
        return $best;
    }
}
