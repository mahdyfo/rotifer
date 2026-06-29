<?php

declare(strict_types=1);

namespace Rotifer\Observe\Reporter;

/** Discards every event. The default, used by tests and quiet runs. */
final class NullReporter implements Reporter
{
    public function onEvent(object $event): void
    {
    }
}
