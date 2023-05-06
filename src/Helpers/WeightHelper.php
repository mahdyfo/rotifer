<?php

namespace GeneticAutoml\Helpers;

class WeightHelper
{
    // Max possible weight for connections. Min possible weight will be -MAX_WEIGHT
    const MAX_WEIGHT = 8.388607; // 1111 1111 1111 1111 1111 1111 / 10^6

    public static function generateRandomWeight(): float
    {
        return mt_rand(-self::MAX_WEIGHT * 1000000, self::MAX_WEIGHT * 1000000) / 1000000;
    }
}