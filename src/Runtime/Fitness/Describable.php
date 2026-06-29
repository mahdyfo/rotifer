<?php

declare(strict_types=1);

namespace Rotifer\Runtime\Fitness;

/**
 * Optional capability: a {@see Problem} that can describe itself in one short
 * sentence, shown beside its name in the dashboard (and `rotifer list`). Like
 * {@see Predictable}, it is deliberately separate from the core Problem
 * interface so existing problems and test doubles need not implement it.
 */
interface Describable
{
    /** A short, one-line summary of the task (plain text, no markup). */
    public function description(): string;
}
