<?php

declare(strict_types=1);

namespace Rotifer\Network\Activation;

/**
 * A per-neuron stand-in for softmax (scaled sigmoid).
 *
 * True softmax needs every neuron in a layer at once, but this engine computes
 * neurons one at a time, so attention-style problems use this monotonic proxy.
 * Named "softmax" for continuity with the transformer example.
 */
final class SoftmaxApprox implements Activation
{
    public function activate(float $x): float
    {
        return 1.0 / (1.0 + exp(-$x * 2.0));
    }

    public function name(): string
    {
        return 'softmax';
    }
}
