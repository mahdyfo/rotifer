<?php

declare(strict_types=1);

namespace Rotifer\Tests\Support;

use Rotifer\Observe\Event\GenerationCompleted;
use Rotifer\Observe\Reporter\Reporter;

/** Records every event so tests can assert on the emitted stream. */
final class CapturingReporter implements Reporter
{
    /** @var list<object> */
    public array $events = [];

    public function onEvent(object $event): void
    {
        $this->events[] = $event;
    }

    /** @return list<float> best fitness per generation, in order */
    public function bestFitnessSeries(): array
    {
        $series = [];
        foreach ($this->events as $event) {
            if ($event instanceof GenerationCompleted) {
                $series[] = $event->bestFitness;
            }
        }
        return $series;
    }
}
