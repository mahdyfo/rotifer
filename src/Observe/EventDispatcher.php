<?php

declare(strict_types=1);

namespace Rotifer\Observe;

use Rotifer\Observe\Reporter\Reporter;

/**
 * Fans engine events out to every registered reporter. Deliberately tiny - the
 * engine depends on this, not on any concrete output.
 */
final class EventDispatcher
{
    /** @var list<Reporter> */
    private array $reporters = [];

    public function add(Reporter $reporter): self
    {
        $this->reporters[] = $reporter;
        return $this;
    }

    public function dispatch(object $event): void
    {
        foreach ($this->reporters as $reporter) {
            $reporter->onEvent($event);
        }
    }
}
