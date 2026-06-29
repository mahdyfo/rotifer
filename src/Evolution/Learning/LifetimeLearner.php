<?php

declare(strict_types=1);

namespace Rotifer\Evolution\Learning;

use Rotifer\Genome\Gene;
use Rotifer\Genome\Genome;
use Rotifer\Genome\Weight;
use Rotifer\Organism\Organism;
use Rotifer\Runtime\Fitness\Problem;
use Rotifer\Runtime\Fitness\Scorer;
use Rotifer\Runtime\Rng;

/**
 * Within-lifetime learning: a short gradient-free hill-climb on an organism's
 * weights against the problem it faces.
 *
 * The fitness an organism competes with is its *learned* score (the Baldwin
 * effect - learning ability is selected even when it isn't inherited). A
 * configurable fraction of what was learned is then written back into the
 * genome, so it *is* inherited (Lamarckian). With fraction 0 you get pure
 * Baldwin; with 1 the full lesson is passed on.
 */
final class LifetimeLearner
{
    public function __construct(
        private readonly Rng $rng,
        private readonly int $steps,
        private readonly float $stepSize,
        private readonly float $lamarckianFraction,
    ) {
    }

    public function refine(Organism $organism, Problem $problem): void
    {
        $spec = $organism->spec();
        $base = $organism->genome();

        $bestGenome = $base;
        $bestFitness = Scorer::score($organism, $problem);

        $genes = $base->genes();
        if ($genes !== []) {
            for ($i = 0; $i < $this->steps; $i++) {
                $candidate = $this->perturb($bestGenome, $this->rng);
                $fitness = Scorer::scoreGenome($candidate, $spec, $problem);
                if ($fitness > $bestFitness) {
                    $bestFitness = $fitness;
                    $bestGenome = $candidate;
                }
            }
        }

        // Lamarckian write-back: inherit only a fraction of the learned change.
        $inherited = $this->lamarckianFraction <= 0.0
            ? $base
            : $this->blend($base, $bestGenome, $this->lamarckianFraction);

        $organism->adoptGenome($inherited)->setFitness($bestFitness);
    }

    private function perturb(Genome $genome, Rng $rng): Genome
    {
        $genes = $genome->genes();
        $gene = $genes[$rng->intBetween(0, count($genes) - 1)];
        $nudged = Weight::clamp($gene->weight + $rng->gaussian(0.0, $this->stepSize));
        return $genome->with($gene->withWeight($nudged));
    }

    /** Move each shared gene's weight from base toward learned by $fraction. */
    private function blend(Genome $base, Genome $learned, float $fraction): Genome
    {
        return $base->map(function (Gene $gene) use ($learned, $fraction): Gene {
            $learnedGene = $learned->get($gene->connectionKey());
            if ($learnedGene === null) {
                return $gene;
            }
            $weight = $gene->weight + $fraction * ($learnedGene->weight - $gene->weight);
            return $gene->withWeight($weight);
        });
    }
}
