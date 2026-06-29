<?php

declare(strict_types=1);

namespace Rotifer\Network\Activation;

/** Classic logistic squashing into (0, 1). The default activation. */
final class Sigmoid implements Activation
{
    public function activate(float $x): float
    {
        return 1.0 / (1.0 + exp(-$x));
    }

    public function name(): string
    {
        return 'sigmoid';
    }
}
