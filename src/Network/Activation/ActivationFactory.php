<?php

declare(strict_types=1);

namespace Rotifer\Network\Activation;

use InvalidArgumentException;

/**
 * Rebuilds an Activation from its name. Used when a config crosses a process
 * boundary (parallel workers) or is read back from a saved run.
 */
final class ActivationFactory
{
    public static function fromName(string $name): Activation
    {
        return match ($name) {
            'sigmoid' => new Sigmoid(),
            'relu' => new Relu(),
            'leaky_relu' => new LeakyRelu(),
            'tanh' => new Tanh(),
            'threshold' => new Threshold(),
            'gelu' => new Gelu(),
            'softmax' => new SoftmaxApprox(),
            default => throw new InvalidArgumentException("Unknown activation: {$name}"),
        };
    }
}
