<?php

declare(strict_types=1);

namespace Rotifer\Runtime\Fitness;

use Rotifer\Runtime\EvolutionConfig;
use Rotifer\Runtime\Rng;

/**
 * Chooses the {@see ScoringWindow} for a generation.
 *
 * The choice is a pure function of (run seed, generation number): every organism
 * in a generation is scored on the *same* window (a fair comparison), the window
 * moves each generation, and the whole run replays exactly under the same seed.
 * Crucially it draws from its own throwaway {@see Rng}, never the island streams,
 * so turning the feature on doesn't perturb the rest of evolution's randomness.
 *
 * Both engines build one the same way ({@see fromConfig}) and ask for the same
 * generation, so the serial {@see \Rotifer\Evolution\World} and the parallel
 * {@see \Rotifer\Runtime\Parallel\IslandEpochTask} agree on the window.
 */
final class WindowSelector
{
    public function __construct(
        private readonly int $seed,
        private readonly int $windowSize,
        private readonly int $prime,
    ) {
    }

    /** Null when the problem doesn't ask for a random window. */
    public static function fromConfig(EvolutionConfig $config): ?self
    {
        if (!$config->isRandomWindowEnabled()) {
            return null;
        }
        return new self($config->getSeed(), $config->getWindowSize(), $config->getWindowPrime());
    }

    /**
     * The window to score generation $generation on, or null when it would cover
     * every scorable row anyway (nothing to randomise, so score the lot).
     *
     * The scored window starts no earlier than `prime` (so the priming rows fit),
     * and the prime is clamped to what actually fits before the window when the
     * data is too short to hold the full prime + window.
     */
    public function forGeneration(int $generation, int $scorableRows): ?ScoringWindow
    {
        if ($this->windowSize <= 0 || $scorableRows <= $this->windowSize) {
            return null;
        }
        $maxStart = $scorableRows - $this->windowSize;
        $minStart = min($this->prime, $maxStart);
        // A fresh stream keyed by the generation: deterministic, and independent of
        // every other Rng in the run so the evolution itself is left untouched.
        $start = (new Rng($this->seed))->derive($generation + 1)->intBetween($minStart, $maxStart);
        return new ScoringWindow($start, $this->windowSize, min($this->prime, $start));
    }

    /**
     * The number of scorable rows in a problem's data: every non-empty row (an
     * empty row is a memory-reset marker, not something to score).
     *
     * @param list<array{0: list<float>, 1: list<float>}|array{}> $data
     */
    public static function countScorable(array $data): int
    {
        $count = 0;
        foreach ($data as $row) {
            if ($row !== []) {
                $count++;
            }
        }
        return $count;
    }
}
