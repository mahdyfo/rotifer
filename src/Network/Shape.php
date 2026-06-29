<?php

declare(strict_types=1);

namespace Rotifer\Network;

/** How many inputs and outputs a problem's networks expose. */
final readonly class Shape
{
    public function __construct(
        public int $inputs,
        public int $outputs,
    ) {
    }
}
