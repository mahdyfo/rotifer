<?php

declare(strict_types=1);

namespace Rotifer\Runtime\Fitness;

/**
 * A randomly-placed slice of a problem's scorable rows to evaluate this generation.
 *
 * When memory is on, always feeding a fixed sequence lets a network cheat by
 * counting steps ("at step 7 output X") instead of using what it remembers.
 * Scoring a randomly-placed window each generation breaks that: the absolute
 * position of a scored row keeps moving, so a step-counter no longer lines up.
 *
 * The network starts cold {@see prime} rows before {@see $start}: rows before
 * `start - prime` are skipped entirely, the `prime` rows in `[start-prime, start)`
 * are fed unscored to build memory context, rows in `[start, start+length)` are
 * scored, and rows after the window are dropped. {@see WindowSelector} picks the
 * start (never earlier than `prime`); the {@see Scorer} walks it.
 */
final class ScoringWindow
{
    public function __construct(
        public readonly int $start,   // index (over scorable rows) of the first scored row
        public readonly int $length,  // how many scorable rows are scored
        public readonly int $prime,   // rows fed unscored immediately before $start (<= $start)
    ) {
    }

    /** The first row fed at all - the cold start, `prime` rows before the window. */
    public function primeStart(): int
    {
        return $this->start - $this->prime;
    }

    /** One past the last scored row. */
    public function end(): int
    {
        return $this->start + $this->length;
    }
}
