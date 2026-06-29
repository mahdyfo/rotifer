<?php

declare(strict_types=1);

namespace Rotifer\Evolution\Reproduction;

use Rotifer\Genome\Gene;
use Rotifer\Genome\Genome;
use Rotifer\Runtime\Rng;

/**
 * Gene-aligned (NEAT-style) crossover of two parent genomes.
 *
 * Genes are matched by connection identity:
 *  - a connection both parents share is inherited from one of them, choosing the
 *    second parent's weight with probability $secondParentBias;
 *  - a connection only one parent has (disjoint/excess) is inherited from the
 *    fitter parent, so structure flows downhill from the better solution. When
 *    fitness ties, disjoint genes from both parents are kept (a union).
 */
final class Crossover
{
    public function cross(
        Genome $first,
        float $firstFitness,
        Genome $second,
        float $secondFitness,
        float $secondParentBias,
        Rng $rng,
    ): Genome {
        $firstByKey = $this->byKey($first);
        $secondByKey = $this->byKey($second);

        $firstWins = $firstFitness >= $secondFitness;
        $secondWins = $secondFitness >= $firstFitness;

        $child = [];
        foreach ($firstByKey as $key => $gene) {
            if (isset($secondByKey[$key])) {
                $child[$key] = $rng->chance($secondParentBias) ? $secondByKey[$key] : $gene;
            } elseif ($firstWins) {
                $child[$key] = $gene; // disjoint, inherited from the (>=) fitter first parent
            }
        }
        foreach ($secondByKey as $key => $gene) {
            if (!isset($firstByKey[$key]) && $secondWins) {
                $child[$key] = $gene; // disjoint, from the (>=) fitter second parent
            }
        }

        return new Genome(array_values($child));
    }

    /** @return array<string, Gene> */
    private function byKey(Genome $genome): array
    {
        $map = [];
        foreach ($genome->genes() as $gene) {
            $map[$gene->connectionKey()] = $gene;
        }
        return $map;
    }
}
