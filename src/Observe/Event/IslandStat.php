<?php

declare(strict_types=1);

namespace Rotifer\Observe\Event;

/**
 * A snapshot of one island in a generation, for the island map. Carries the
 * fields the biology phases populate; until then a run reports a single island
 * covering the whole population.
 */
final readonly class IslandStat
{
    public function __construct(
        public int $index,
        public int $size,
        public float $bestFitness,
        public float $averageFitness,
        public float $mutationScale = 1.0,
        public float $traumaLevel = 0.0,
    ) {
    }
}
