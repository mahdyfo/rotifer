<?php

declare(strict_types=1);

namespace Rotifer\Network\Activation;

/**
 * A neuron activation function. Implementations are tiny, stateless and
 * reconstructable by name so a Problem's config can be rebuilt inside a worker.
 */
interface Activation
{
    public function activate(float $x): float;

    /** Stable identifier, e.g. "sigmoid", used in serialization and the UI. */
    public function name(): string;
}
