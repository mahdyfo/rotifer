<?php

declare(strict_types=1);

namespace Rotifer\Runtime;

/**
 * A small, seedable pseudo-random generator with its own private state.
 *
 * The whole framework draws randomness through an Rng instance instead of the
 * global mt_rand(), which is what makes a run reproducible (same seed => same
 * evolution) and lets each island own an independent, deterministic stream for
 * parallel execution. The algorithm is a 32-bit xorshift: fast, dependency-free,
 * and more than good enough for a genetic search.
 */
final class Rng
{
    private const MASK32 = 0xFFFFFFFF;

    private int $state;
    private ?float $gaussianSpare = null;

    public function __construct(private readonly int $seed)
    {
        // xorshift must never be seeded with 0; fold the seed into a non-zero 32-bit state.
        $s = ($seed ^ 0x9E3779B9) & self::MASK32;
        $this->state = $s !== 0 ? $s : 0x1A2B3C4D;
    }

    public function seed(): int
    {
        return $this->seed;
    }

    /**
     * Spawn an independent child generator for a sub-stream (e.g. an island).
     * Deterministic in the parent's seed, so the whole tree of streams replays.
     */
    public function derive(int $stream): self
    {
        $mixed = ($this->seed * 0x2545F491 + $stream * 0x9E3779B1 + 0x6C078965) & PHP_INT_MAX;
        return new self($mixed);
    }

    /** Raw 32-bit draw. */
    public function nextUint32(): int
    {
        $x = $this->state;
        $x ^= ($x << 13) & self::MASK32;
        $x ^= $x >> 17;
        $x ^= ($x << 5) & self::MASK32;
        $x &= self::MASK32;
        $this->state = $x;
        return $x;
    }

    /** Float in [0, 1). */
    public function nextFloat(): float
    {
        return $this->nextUint32() / 4294967296.0;
    }

    /** Integer in [$min, $max] inclusive. */
    public function intBetween(int $min, int $max): int
    {
        if ($min > $max) {
            [$min, $max] = [$max, $min];
        }
        $span = $max - $min + 1;
        return $min + (int) ($this->nextFloat() * $span);
    }

    /** Float in [$min, $max). */
    public function floatBetween(float $min, float $max): float
    {
        return $min + ($max - $min) * $this->nextFloat();
    }

    /** True with probability $p (clamped to [0,1]). */
    public function chance(float $p): bool
    {
        if ($p <= 0.0) {
            return false;
        }
        if ($p >= 1.0) {
            return true;
        }
        return $this->nextFloat() < $p;
    }

    /**
     * A random existing key of $array.
     * @param array<array-key, mixed> $array
     * @return array-key
     */
    public function pickKey(array $array): string|int
    {
        $keys = array_keys($array);
        return $keys[$this->intBetween(0, count($keys) - 1)];
    }

    /**
     * A random value of $array.
     * @param array<array-key, mixed> $array
     */
    public function pick(array $array): mixed
    {
        return $array[$this->pickKey($array)];
    }

    /**
     * A deterministic Fisher-Yates shuffle returning a new, reindexed list.
     * @param array<array-key, mixed> $array
     * @return list<mixed>
     */
    public function shuffle(array $array): array
    {
        $values = array_values($array);
        for ($i = count($values) - 1; $i > 0; $i--) {
            $j = $this->intBetween(0, $i);
            [$values[$i], $values[$j]] = [$values[$j], $values[$i]];
        }
        return $values;
    }

    /** A normally distributed value (Box-Muller, cached spare). */
    public function gaussian(float $mean = 0.0, float $stdDev = 1.0): float
    {
        if ($this->gaussianSpare !== null) {
            $z = $this->gaussianSpare;
            $this->gaussianSpare = null;
            return $mean + $stdDev * $z;
        }
        do {
            $u = $this->nextFloat() * 2 - 1;
            $v = $this->nextFloat() * 2 - 1;
            $s = $u * $u + $v * $v;
        } while ($s >= 1.0 || $s === 0.0);

        $factor = sqrt(-2.0 * log($s) / $s);
        $this->gaussianSpare = $v * $factor;
        return $mean + $stdDev * ($u * $factor);
    }

    /** A random weight within the legal connection-weight range. */
    public function weight(): float
    {
        return $this->floatBetween(-\Rotifer\Genome\Weight::MAX, \Rotifer\Genome\Weight::MAX);
    }
}
