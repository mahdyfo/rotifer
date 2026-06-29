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
    /** @param list<Organism> $organisms */
    public function evaluate(array $organisms, Problem $problem): void;
}
