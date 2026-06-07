<?php

namespace Rotifer\Activations;

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

    public static function leakyRelu(float $value): float
    {
        return max(0.01 * $value, $value);
    }

    public static function tanh(float $value, float $multiplication = 0.25): float
    {
        return tanh($value * $multiplication);
    }

    public static function threshold(float $value): float
    {
        return $value < 0 ? 0 : 1;
    }

    /**
     * Softmax activation - used for attention mechanisms
     * Note: This is a single-value approximation. For true softmax,
     * you need all values in the layer, but Rotifer processes neurons individually.
     * This approximation uses a scaled sigmoid as a proxy.
     */
    public static function softmax(float $value): float
    {
        // Approximate softmax with scaled sigmoid
        // This isn't true softmax but works within Rotifer's architecture
        return self::sigmoid($value * 2);
    }

    /**
     * GELU (Gaussian Error Linear Unit) - commonly used in transformers
     */
    public static function gelu(float $value): float
    {
        return 0.5 * $value * (1 + tanh(sqrt(2 / M_PI) * ($value + 0.044715 * pow($value, 3))));
    }
}