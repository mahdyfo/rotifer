<?php

declare(strict_types=1);

namespace Rotifer\Network\Activation;

/** ReLU that leaks a small slope for negative inputs, avoiding dead neurons. */
final class LeakyRelu implements Activation
{
    public function __construct(private readonly float $leak = 0.01)
    {
    }

    public function activate(float $x): float
    {
        return $x > 0.0 ? $x : $this->leak * $x;
    }

    public function name(): string
    {
        return 'leaky_relu';
    }
}
