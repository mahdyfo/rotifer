<?php

declare(strict_types=1);

namespace Rotifer\Network\Activation;

/** Rectified linear unit: max(0, x). */
final class Relu implements Activation
{
    public function activate(float $x): float
    {
        return $x > 0.0 ? $x : 0.0;
    }

    public function name(): string
    {
        return 'relu';
    }
}
