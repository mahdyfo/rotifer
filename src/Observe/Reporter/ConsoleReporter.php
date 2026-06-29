<?php

declare(strict_types=1);

namespace Rotifer\Observe\Reporter;

use Rotifer\Observe\Event\GenerationCompleted;
use Rotifer\Observe\Event\RunEnded;
use Rotifer\Observe\Event\RunStarted;

/**
 * A plain one-line-per-generation log to stdout - the no-frills successor to the
 * legacy printf. The rich live terminal dashboard is a separate reporter.
 */
final class ConsoleReporter implements Reporter
{
    public function __construct(private readonly bool $quiet = false)
    {
    }

    public function onEvent(object $event): void
    {
        if ($this->quiet) {
            return;
        }

        match (true) {
            $event instanceof RunStarted => $this->line("Evolving \"{$event->problemName}\" (seed {$event->config->getSeed()}, pop {$event->config->getPopulation()})"),
            $event instanceof GenerationCompleted => $this->onGeneration($event),
            $event instanceof RunEnded => $this->line(sprintf('Done: best fitness %.6f over %d generations', $event->bestFitness, $event->generationsRun)),
            default => null,
        };
    }

    private function onGeneration(GenerationCompleted $e): void
    {
        $this->line(sprintf(
            '[%s] Gen %4d | best %.6f (all-time %.6f) | avg %.6f | hidden %d | genes %d | %.3fs%s',
            date('H:i:s'),
            $e->generation,
            $e->bestFitness,
            $e->allTimeBestFitness,
            $e->averageFitness,
            $e->bestHiddenCount,
            $e->bestGeneCount,
            $e->durationSeconds,
            $e->improved ? '  [NEW BEST]' : '',
        ));
    }

    private function line(string $text): void
    {
        fwrite(STDOUT, $text . PHP_EOL);
    }
}
