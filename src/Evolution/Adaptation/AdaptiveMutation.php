<?php

declare(strict_types=1);

namespace Rotifer\Evolution\Adaptation;

/**
 * Self-tuning mutation pressure for one island.
 *
 * While the island keeps finding new bests it eases off (exploit); once it
 * stalls for `patience` generations it ramps mutation up (explore), then resets
 * the patience clock. The result is a scale factor the mutator multiplies its
 * rates by - bounded so it never collapses to zero or explodes.
 */
final class AdaptiveMutation
{
    private const EPSILON = 1e-9;

    private float $scale = 1.0;
    private float $bestSoFar = -INF;
    private int $stagnation = 0;

    public function __construct(
        private readonly int $patience,
        private readonly float $upFactor,
        private readonly float $downFactor,
        private readonly float $minScale,
        private readonly float $maxScale,
    ) {
    }

    public function update(float $generationBest): float
    {
        if ($generationBest > $this->bestSoFar + self::EPSILON) {
            $this->bestSoFar = $generationBest;
            $this->stagnation = 0;
            $this->scale *= $this->downFactor;
        } else {
            $this->stagnation++;
            if ($this->stagnation >= $this->patience) {
                $this->scale *= $this->upFactor;
                $this->stagnation = 0;
            }
        }

        $this->scale = min($this->maxScale, max($this->minScale, $this->scale));
        return $this->scale;
    }

    public function scale(): float
    {
        return $this->scale;
    }

    /**
     * The evolving state (not the fixed knobs), so a stopped run can resume with
     * the same mutation pressure instead of snapping back to the default.
     *
     * @return array{scale: float, bestSoFar: float, stagnation: int}
     */
    public function state(): array
    {
        return ['scale' => $this->scale, 'bestSoFar' => $this->bestSoFar, 'stagnation' => $this->stagnation];
    }

    /** @param array{scale?: float, bestSoFar?: float, stagnation?: int} $state */
    public function restore(array $state): void
    {
        $this->scale = (float) ($state['scale'] ?? $this->scale);
        // -INF does not survive JSON; a null/missing best means "nothing seen yet".
        $this->bestSoFar = isset($state['bestSoFar']) ? (float) $state['bestSoFar'] : -INF;
        $this->stagnation = (int) ($state['stagnation'] ?? $this->stagnation);
    }
}
