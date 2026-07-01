<?php

declare(strict_types=1);

namespace Rotifer\Runtime\Fitness;

use Rotifer\Evolution\Learning\LifetimeLearner;

/**
 * Evaluates organisms one after another in this process. The reference
 * implementation: deterministic, dependency-free, and what tests assert against
 * (the parallel evaluator must match it bit for bit).
 *
 * If a {@see LifetimeLearner} is supplied, each organism is refined within its
 * lifetime before its fitness is recorded (Baldwin effect, optionally writing
 * gains back into the genome for Lamarckian inheritance); otherwise fitness is
 * the plain score from {@see Scorer}.
 */
final class SerialEvaluator implements FitnessEvaluator
{
    public function __construct(private readonly ?LifetimeLearner $learner = null)
    {
    }

    public function evaluate(array $organisms, Problem $problem, ?ScoringWindow $window = null): void
    {
        foreach ($organisms as $organism) {
            if ($this->learner !== null) {
                $this->learner->refine($organism, $problem, $window);
            } else {
                $organism->setFitness(Scorer::score($organism, $problem, $window));
            }
        }
    }
}
