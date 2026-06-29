<?php

declare(strict_types=1);

namespace Rotifer\Observe\Reporter;

use Rotifer\Observe\Event\GenerationCompleted;
use Rotifer\Observe\Event\IslandStat;
use Rotifer\Observe\Event\RunEnded;
use Rotifer\Observe\Event\RunStarted;
use Rotifer\Persistence\Codec\JsonCodec;
use Rotifer\Persistence\SnapshotStore;
use Rotifer\Runtime\FastRuntime;

/**
 * Writes the run as it happens to runs/<name>/: one JSON object per generation
 * appended to stream.jsonl, plus meta.json and best.json. This file *is* the
 * contract with the web dashboard - the server tails stream.jsonl over SSE and
 * the browser renders it, and replays read the same files back.
 */
final class JsonStreamReporter implements Reporter
{
    private string $name = 'run';
    private int $inputs = 0;
    private int $outputs = 0;
    /** @var list<int> fixed hidden-layer sizes, or [] for dynamic topology */
    private array $hiddenLayers = [];
    /** wall-clock start of this run instance, and the elapsed total a resume carries over */
    private float $startMicrotime = 0.0;
    private float $elapsedBase = 0.0;

    public function __construct(
        private readonly SnapshotStore $store = new SnapshotStore(),
        private readonly JsonCodec $codec = new JsonCodec(),
    ) {
    }

    public function onEvent(object $event): void
    {
        match (true) {
            $event instanceof RunStarted => $this->begin($event),
            $event instanceof GenerationCompleted => $this->append($event),
            $event instanceof RunEnded => $this->finish($event),
            default => null,
        };
    }

    private function begin(RunStarted $event): void
    {
        $this->name = $event->problemName;
        $this->inputs = $event->inputs;
        $this->outputs = $event->outputs;
        $this->hiddenLayers = $event->config->getHiddenLayers();

        $this->store->ensure($this->name);
        // A resume keeps appending to the existing stream; a fresh run truncates it.
        if (!$event->resume) {
            file_put_contents($this->store->streamPath($this->name), '');
        }
        // The "elapsed" clock measures the whole run: a fresh run starts at zero, a
        // resume continues from the total the previous session reached.
        $this->startMicrotime = microtime(true);
        $this->elapsedBase = $event->resume ? $this->lastElapsedMs() : 0.0;

        $config = $event->config;
        $meta = is_file($this->store->metaPath($this->name))
            ? (json_decode((string) file_get_contents($this->store->metaPath($this->name)), true) ?: [])
            : [];
        // When resuming, the run adds its generations on top of what was already done.
        $generations = $event->resume
            ? (int) ($meta['generations'] ?? 0) + $config->getGenerations()
            : $config->getGenerations();

        $this->writeJson($this->store->metaPath($this->name), [
            'problem' => $event->problemName,
            'seed' => $config->getSeed(),
            'population' => $config->getPopulation(),
            'islands' => $config->getIslands(),
            'generations' => $generations,
            'activation' => $config->getActivation()->name(),
            'inputs' => $this->inputs,
            'outputs' => $this->outputs,
            'memory' => $config->hasMemory(),
            'startedAt' => $event->resume ? ($meta['startedAt'] ?? date('c')) : date('c'),
            // A fresh run gets a new id (it truncates and rewrites the stream); a
            // resume keeps the id, since it just appends. The dashboard uses this to
            // tell "the run restarted, follow it from the top" from "more of the same".
            'runId' => $event->resume ? ($meta['runId'] ?? $this->newRunId()) : $this->newRunId(),
            // The run's own runtime health (JIT on? Xdebug slowing?), so the dashboard
            // can warn if this run did not get the speed-up.
            'runtime' => FastRuntime::diagnostics(),
        ]);
    }

    private function append(GenerationCompleted $event): void
    {
        $line = json_encode([
            'gen' => $event->generation,
            'totalGenerations' => $event->totalGenerations,
            'best' => $event->bestFitness,
            'avg' => $event->averageFitness,
            'allTimeBest' => $event->allTimeBestFitness,
            'popSize' => $event->populationSize,
            'hidden' => $event->bestHiddenCount,
            'genes' => $event->bestGeneCount,
            'improved' => $event->improved,
            'matchRate' => $event->matchRate,
            'durationMs' => round($event->durationSeconds * 1000, 2),
            'elapsedMs' => round($this->elapsedBase + (microtime(true) - $this->startMicrotime) * 1000),
            'islands' => array_map($this->islandToArray(...), $event->islands),
            'network' => [
                'inputs' => $this->inputs,
                'outputs' => $this->outputs,
                'layers' => $this->hiddenLayers, // hidden-layer sizes; [] = dynamic
                'genes' => json_decode($this->codec->encode($event->bestGenome), true),
            ],
        ], JSON_THROW_ON_ERROR);

        file_put_contents($this->store->streamPath($this->name), $line . "\n", FILE_APPEND);
    }

    private function finish(RunEnded $event): void
    {
        $this->writeJson($this->store->bestPath($this->name), [
            'problem' => $event->problemName,
            'generations' => $event->generationsRun,
            'bestFitness' => $event->bestFitness,
            'finishedAt' => date('c'),
        ]);
        $this->writeJson($this->store->predictionsPath($this->name), $event->predictions);
    }

    private function islandToArray(IslandStat $stat): array
    {
        return [
            'index' => $stat->index,
            'size' => $stat->size,
            'best' => $stat->bestFitness,
            'avg' => $stat->averageFitness,
            'mutationScale' => $stat->mutationScale,
            'trauma' => $stat->traumaLevel,
        ];
    }

    private function newRunId(): string
    {
        return bin2hex(random_bytes(6));
    }

    /** The elapsed total the existing stream reached, so a resume continues the clock. */
    private function lastElapsedMs(): float
    {
        $path = $this->store->streamPath($this->name);
        if (!is_file($path)) {
            return 0.0;
        }
        $lines = array_values(array_filter(explode("\n", (string) file_get_contents($path))));
        $last = end($lines);
        if ($last === false) {
            return 0.0;
        }
        $record = json_decode($last, true);
        return (float) ($record['elapsedMs'] ?? 0.0);
    }

    private function writeJson(string $path, array $data): void
    {
        file_put_contents($path, json_encode($data, JSON_PRETTY_PRINT | JSON_THROW_ON_ERROR));
    }
}
