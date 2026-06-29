<?php

declare(strict_types=1);

namespace Rotifer\Network\Activation;

/** Hard step: 0 for negative input, 1 otherwise. */
final class Threshold implements Activation
{
    public function activate(float $x): float
    {
        return $x < 0.0 ? 0.0 : 1.0;
    }

    public function name(): string
    {
        return 'threshold';
    }
}
