<?php

declare(strict_types=1);

namespace Rotifer\Genome;

/**
 * The single source of truth for connection-weight bounds and precision.
 *
 * The limit comes from the binary/hex codec: a weight is stored as a signed
 * 24-bit integer scaled by 1e6, giving a usable range of +-8.388607 with six
 * decimal places. Mutation, codecs and config all clamp through here (DRY).
 */
final class Weight
{
    /** 2^23 - 1, the largest magnitude representable in 24 signed bits. */
    public const SCALE = 1_000_000;
    public const MAX = 8.388607; // (2^23 - 1) / SCALE

    public static function clamp(float $weight): float
    {
        if ($weight > self::MAX) {
            return self::MAX;
        }
        if ($weight < -self::MAX) {
            return -self::MAX;
        }
        return $weight;
    }

    public static function isInRange(float $weight): bool
    {
        return $weight >= -self::MAX && $weight <= self::MAX;
    }
}
