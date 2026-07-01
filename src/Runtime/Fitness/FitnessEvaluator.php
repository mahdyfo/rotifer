<?php

declare(strict_types=1);

namespace Rotifer\Runtime\Fitness;

use Rotifer\Organism\Organism;

/**
 * Scores a batch of organisms against a problem, writing each organism's total
 * fitness onto it. The serial and parallel engines are interchangeable behind
 * this interface.
 */
interface FitnessEvaluator
{
    /**
     * @param list<Organism> $organisms
     * @param ?ScoringWindow $window this generation's scoring window, or null to score every row
     */
    public function evaluate(array $organisms, Problem $problem, ?ScoringWindow $window = null): void;
}
