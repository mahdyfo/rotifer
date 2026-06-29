<?php

declare(strict_types=1);

namespace Rotifer\Observe\Reporter;

use Rotifer\Observe\Event\GenerationCompleted;
use Rotifer\Observe\Event\IslandStat;
use Rotifer\Observe\Event\RunEnded;
use Rotifer\Observe\Event\RunStarted;

/**
 * A live, self-redrawing terminal panel: a Unicode sparkline of best fitness, a
 * per-island/species table, and a compact sketch of the champion network.
 *
 * Each generation it moves the cursor back over the previous panel and repaints,
 * so the dashboard updates in place rather than scrolling - the rich successor
 * to the legacy single printf line. (ANSI throughout; "\e" is the escape byte.)
 */
final class TerminalDashboard implements Reporter
{
    private const BLOCKS = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    private const SPARK_WIDTH = 60;

    /** @var list<float> */
    private array $history = [];
    private int $lastLines = 0;
    private int $inputs = 0;
    private int $outputs = 0;
    private float $startedAt = 0.0;

    public function onEvent(object $event): void
    {
        match (true) {
            $event instanceof RunStarted => $this->begin($event),
            $event instanceof GenerationCompleted => $this->repaint($event),
            $event instanceof RunEnded => $this->end($event),
            default => null,
        };
    }

    private function begin(RunStarted $event): void
    {
        $this->inputs = $event->inputs;
        $this->outputs = $event->outputs;
        $this->startedAt = microtime(true);

        $c = $event->config;
        $bio = array_filter([
            $c->isTraumaEnabled() ? 'trauma' : null,
            $c->isAdaptiveMutationEnabled() ? 'adaptive-mut' : null,
            $c->isLifetimeLearningEnabled() ? 'learning' : null,
            $c->getIslands() > 1 ? "{$c->getIslands()} islands" : null,
        ]);

        $this->write(sprintf(
            "\n  \e[1mRotifer\e[0m - evolving \e[36m%s\e[0m  (seed %d, pop %d%s)\n\n",
            $event->problemName,
            $c->getSeed(),
            $c->getPopulation(),
            $bio !== [] ? ', ' . implode(' · ', $bio) : '',
        ));
    }

    private function repaint(GenerationCompleted $event): void
    {
        $this->history[] = $event->bestFitness;
        if (count($this->history) > self::SPARK_WIDTH) {
            $this->history = array_slice($this->history, -self::SPARK_WIDTH);
        }

        $progress = $event->totalGenerations > 0
            ? sprintf('%d/%d', $event->generation, $event->totalGenerations)
            : (string) $event->generation;

        $lines = [];
        $lines[] = sprintf(
            "  gen %-9s  best \e[32m%.4f\e[0m   avg %.4f   all-time \e[1m%.4f\e[0m%s",
            $progress,
            $event->bestFitness,
            $event->averageFitness,
            $event->allTimeBestFitness,
            $event->improved ? "  \e[33m★\e[0m" : '',
        );
        $lines[] = '  fitness ' . $this->sparkline($this->history);
        $lines[] = sprintf(
            "  network \e[34mIN(%d)\e[0m ─▶ \e[35mH(%d)\e[0m ─▶ \e[34mOUT(%d)\e[0m   genes %d   %.0f ms/gen",
            $this->inputs,
            $event->bestHiddenCount,
            $this->outputs,
            $event->bestGeneCount,
            $event->durationSeconds * 1000,
        );
        $lines[] = '';
        foreach ($this->islandTable($event->islands) as $row) {
            $lines[] = $row;
        }

        $this->paint($lines);
    }

    private function end(RunEnded $event): void
    {
        $this->write(sprintf(
            "\n  \e[1mDone\e[0m - best fitness \e[32m%.6f\e[0m over %d generations in %.1fs\n",
            $event->bestFitness,
            $event->generationsRun,
            microtime(true) - $this->startedAt,
        ));
    }

    /** @param list<float> $values */
    private function sparkline(array $values): string
    {
        if ($values === []) {
            return '';
        }
        $min = min($values);
        $max = max($values);
        $span = ($max - $min) ?: 1.0;
        $bars = '';
        foreach ($values as $value) {
            $level = (int) round(($value - $min) / $span * (count(self::BLOCKS) - 1));
            $bars .= self::BLOCKS[$level];
        }
        return "\e[36m{$bars}\e[0m";
    }

    /**
     * @param list<IslandStat> $islands
     * @return list<string>
     */
    private function islandTable(array $islands): array
    {
        // A plain single deme with no active biology has nothing worth tabulating.
        if (count($islands) <= 1
            && ($islands[0]->traumaLevel ?? 0.0) <= 0.0
            && ($islands[0]->mutationScale ?? 1.0) === 1.0
        ) {
            return [];
        }

        $rows = ["  \e[2misland   size   best      mut×   trauma\e[0m"];
        foreach ($islands as $island) {
            $rows[] = sprintf(
                '  %-7d  %-4d   %-8.4f  %-4.2f   %.2f',
                $island->index,
                $island->size,
                $island->bestFitness,
                $island->mutationScale,
                $island->traumaLevel,
            );
        }
        $rows[] = '';
        return $rows;
    }

    /** @param list<string> $lines */
    private function paint(array $lines): void
    {
        $out = '';
        if ($this->lastLines > 0) {
            $out .= "\e[{$this->lastLines}F\e[0J"; // back up over the old panel and clear it
        }
        $out .= implode("\n", $lines) . "\n";
        $this->write($out);
        $this->lastLines = count($lines);
    }

    private function write(string $text): void
    {
        fwrite(STDOUT, $text);
    }
}
