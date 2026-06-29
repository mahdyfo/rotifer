<?php

declare(strict_types=1);

namespace Rotifer\Network;

use Rotifer\Network\Activation\Activation;

/**
 * The run-wide constants every Brain needs but no genome carries: I/O shape,
 * whether hidden state persists between steps (memory), and which activation to
 * apply. The engine builds one of these from the Problem's shape and config and
 * hands the same instance to every organism.
 */
final readonly class NetworkSpec
{
    public function __construct(
        public Shape $shape,
        public bool $hasMemory,
        public Activation $activation,
        // When set, the network is a fixed layered MLP (see {@see LayerPlan});
        // null keeps Rotifer's default dynamic, evolving topology.
        public ?LayerPlan $layers = null,
    ) {
    }

    public function inputs(): int
    {
        return $this->shape->inputs;
    }

    public function outputs(): int
    {
        return $this->shape->outputs;
    }

    /** True when this run uses a fixed layered topology instead of evolving one. */
    public function isLayered(): bool
    {
        return $this->layers !== null && $this->layers->layerCount() > 0;
    }
}
