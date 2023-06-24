<?php

namespace GeneticAutoml\Activations;

class Activation
{
    public static function sigmoid(float $value): float
    {
        return 1 / (1 + pow(M_E, -$value));
    }

    public static function relu(float $value): float
    {
        return max(0, $value);
    }

    public static function tanh(float $value): float
    {
        return tanh($value);
    }

    public static function threshold(float $value): float
    {
        return $value < 0 ? 0 : 1;
    }
}