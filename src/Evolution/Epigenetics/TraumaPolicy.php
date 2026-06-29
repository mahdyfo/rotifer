<?php

declare(strict_types=1);

namespace Rotifer\Evolution\Epigenetics;

use Rotifer\Organism\Epigenome;
use Rotifer\Organism\Organism;

/**
 * Models "genetic trauma": hardship leaves a heritable, fading mark that makes
 * descendants mutate more for a few generations.
 *
 * After evaluation, organisms doing worse than their peers are stressed. A child
 * inherits the decayed blend of its parents' marks, and that residual stress
 * scales up the child's mutation rate. Because the mark decays each inheritance,
 * an ancestor's trauma washes out of the lineage over time.
 */
final class TraumaPolicy
{
    public function __construct(
        private readonly float $intensity,
        private readonly float $decay,
    ) {
    }

    /** Stamp below-average performers with stress. @param list<Organism> $organisms */
    public function applyStress(array $organisms): void
    {
        if ($organisms === []) {
            return;
        }
        $average = 0.0;
        foreach ($organisms as $organism) {
            $average += $organism->fitness();
        }
        $average /= count($organisms);

        foreach ($organisms as $organism) {
            if ($organism->fitness() < $average) {
                $organism->epigenome()->raise('stress', $this->intensity);
            }
        }
    }

    /** The decayed, blended epigenome a child of these parents is born with. */
    public function childEpigenome(Organism $first, Organism $second): Epigenome
    {
        $child = Epigenome::inherit($first->epigenome(), $second->epigenome());
        $child->decay($this->decay);
        return $child;
    }

    /** Mutation-rate multiplier implied by an epigenome's residual stress. */
    public function mutationBoost(Epigenome $epigenome): float
    {
        return 1.0 + $epigenome->intensity('stress');
    }
}
