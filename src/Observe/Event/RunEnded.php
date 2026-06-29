<?php

declare(strict_types=1);

namespace Rotifer\Observe\Event;

/** Emitted once, after the final generation. */
final readonly class RunEnded
{
    /**
     * @param array{columns: list<string>, rows: list<list<string>>} $predictions
     *        the champion's results, ready to render as a table
     */
    public function __construct(
        public string $problemName,
        public int $generationsRun,
        public float $bestFitness,
        public array $predictions = ['columns' => [], 'rows' => []],
    ) {
    }
}
