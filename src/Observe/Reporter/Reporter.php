<?php

declare(strict_types=1);

namespace Rotifer\Observe\Reporter;

/**
 * Anything that reacts to engine events - console dashboards, JSON stream
 * writers, snapshot stores. The engine never writes output directly; it emits
 * events and reporters decide what to do, so the same run can feed a terminal,
 * a browser and a replay file at once.
 */
interface Reporter
{
    public function onEvent(object $event): void;
}
