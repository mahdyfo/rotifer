<?php

declare(strict_types=1);

namespace Rotifer\Evolution\Reproduction;

use Rotifer\Network\NetworkSpec;
use Rotifer\Organism\Organism;
use Rotifer\Runtime\EvolutionConfig;
use Rotifer\Runtime\Rng;

/**
 * Turns two parents into one child: crossover, then mutation, then compile.
 *
 * Mutation can occasionally prune a genome down to nothing; rather than throw
 * (as the legacy engine did after 100 tries), this retries a few times and then
 * falls back to the fitter parent's genome - evolution never stalls on a bad
 * dice roll.
 */
final class Reproducer
{
    private const MAX_ATTEMPTS = 8;

    public function __construct(
        private readonly EvolutionConfig $config,
        private readonly NetworkSpec $spec,
        private readonly Rng $rng,
        private readonly Crossover $crossover = new Crossover(),
        private readonly Mutator $mutator = new Mutator(),
    ) {
    }

    public function breed(Organism $first, Organism $second, float $rateScale = 1.0): Organism
    {
        for ($attempt = 0; $attempt < self::MAX_ATTEMPTS; $attempt++) {
            $childGenome = $this->crossover->cross(
                $first->genome(),
                $first->fitness(),
                $second->genome(),
                $second->fitness(),
                $this->config->getCrossoverProbability(),
                $this->rng,
            );
            $childGenome = $this->mutator->mutate($childGenome, $this->spec, $this->config, $this->rng, $rateScale);

            $child = new Organism($childGenome, $this->spec);
            if (!$child->genome()->isEmpty()) {
                return $child;
            }
        }

        $fitter = $first->fitness() >= $second->fitness() ? $first : $second;
        return new Organism($fitter->genome(), $this->spec);
    }
}
