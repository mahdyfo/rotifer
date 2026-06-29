<?php

declare(strict_types=1);

namespace Rotifer\Tests\Unit\Observe;

use PHPUnit\Framework\TestCase;
use Rotifer\Observe\EventDispatcher;
use Rotifer\Tests\Support\CapturingReporter;

final class EventDispatcherTest extends TestCase
{
    public function testDispatchesToEveryReporter(): void
    {
        $a = new CapturingReporter();
        $b = new CapturingReporter();
        $dispatcher = (new EventDispatcher())->add($a)->add($b);

        $event = new \stdClass();
        $dispatcher->dispatch($event);

        $this->assertSame([$event], $a->events);
        $this->assertSame([$event], $b->events);
    }
}
