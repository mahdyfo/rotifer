<?php

declare(strict_types=1);

namespace Rotifer\Network\Activation;

/**
 * Hyperbolic tangent into (-1, 1). The input is damped (default 0.25, matching
 * the legacy engine) so the function stays in its responsive region for the
 * weight ranges this framework produces.
 */
final class Tanh implements Activation
{
    public function __construct(private readonly float $damping = 0.25)
    {
    }

    public function activate(float $x): float
    {
        return tanh($x * $this->damping);
    }

    public function name(): string
    {
        return 'tanh';
    }
}
