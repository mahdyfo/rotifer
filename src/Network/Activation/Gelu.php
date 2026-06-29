<?php

declare(strict_types=1);

namespace Rotifer\Network\Activation;

/** Gaussian Error Linear Unit (tanh approximation), the transformer standard. */
final class Gelu implements Activation
{
    public function activate(float $x): float
    {
        return 0.5 * $x * (1.0 + tanh(sqrt(2.0 / M_PI) * ($x + 0.044715 * $x ** 3)));
    }

    public function name(): string
    {
        return 'gelu';
    }
}
