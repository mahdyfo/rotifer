<?php

declare(strict_types=1);

namespace Rotifer\Evolution\Migration;

/**
 * Decides when and where organisms travel between islands - a ring topology
 * where island i sends its best emigrants to island i+1 every N generations.
 *
 * Just the policy (timing + neighbour + how many); the World performs the actual
 * copying, since only it can mint fresh organism ids. Spreading good genes while
 * keeping demes distinct is what lets different "villages" specialize yet still
 * share breakthroughs.
 */
final class MigrationPolicy
{
    public function __construct(
        private readonly int $everyGenerations,
        private readonly int $topK,
    ) {
    }

    public function isDue(int $generation): bool
    {
        return $this->everyGenerations > 0
            && $generation > 0
            && $generation % $this->everyGenerations === 0;
    }

    /** The island that island $index sends emigrants to (ring neighbour). */
    public function destinationOf(int $index, int $islandCount): int
    {
        return $islandCount > 1 ? ($index + 1) % $islandCount : $index;
    }

    public function topK(): int
    {
        return $this->topK;
    }
}
