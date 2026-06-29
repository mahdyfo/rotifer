<?php

declare(strict_types=1);

namespace Rotifer\Observe\Event;

use Rotifer\Genome\Genome;

/**
 * Emitted after each generation is evaluated and bred. Carries everything the
 * dashboards need; the best genome travels along so the UI can draw the current
 * champion's network without reaching back into the engine.
 *
 * @param list<IslandStat> $islands per-island stats (one entry until islands land)
 */
final readonly class GenerationCompleted
{
    public function __construct(
        public int $generation,
        public int $totalGenerations,
        public float $bestFitness,
        public float $averageFitness,
        public float $allTimeBestFitness,
        public int $populationSize,
        public int $bestHiddenCount,
        public int $bestGeneCount,
        public bool $improved,
        public float $durationSeconds,
        public Genome $bestGenome,
        public array $islands = [],
        public ?float $matchRate = null,
    ) {
    }
}
