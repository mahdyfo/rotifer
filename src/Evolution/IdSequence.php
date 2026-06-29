<?php

declare(strict_types=1);

namespace Rotifer\Evolution;

/**
 * Hands out monotonically increasing organism ids. Shared across all islands so
 * ids are globally unique and assignment order is deterministic - which keeps
 * fitness-tie sorting (and therefore whole runs) reproducible.
 */
final class IdSequence
{
    private int $next = 0;

    public function next(): string
    {
        return (string) $this->next++;
    }
}
