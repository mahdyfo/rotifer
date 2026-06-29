<?php

declare(strict_types=1);

namespace Rotifer\Observe\Event;

use Rotifer\Runtime\EvolutionConfig;

/** Emitted once, before the first generation. */
final readonly class RunStarted
{
    public function __construct(
        public string $problemName,
        public EvolutionConfig $config,
        public int $inputs,
        public int $outputs,
        public bool $resume = false,
    ) {
    }
}
