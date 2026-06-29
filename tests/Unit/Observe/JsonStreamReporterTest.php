<?php

declare(strict_types=1);

namespace Rotifer\Tests\Unit\Observe;

use PHPUnit\Framework\TestCase;
use Rotifer\Genome\Gene;
use Rotifer\Genome\Genome;
use Rotifer\Genome\NodeType;
use Rotifer\Observe\Event\GenerationCompleted;
use Rotifer\Observe\Event\IslandStat;
use Rotifer\Observe\Event\RunEnded;
use Rotifer\Observe\Event\RunStarted;
use Rotifer\Observe\Reporter\JsonStreamReporter;
use Rotifer\Persistence\SnapshotStore;
use Rotifer\Runtime\EvolutionConfig;

final class JsonStreamReporterTest extends TestCase
{
    private string $base;
    private SnapshotStore $store;

    protected function setUp(): void
    {
        $this->base = sys_get_temp_dir() . '/rotifer_test_' . uniqid();
        $this->store = new SnapshotStore($this->base);
    }

    protected function tearDown(): void
    {
        if (is_dir($this->base)) {
            array_map('unlink', glob($this->base . '/*/*') ?: []);
            array_map('rmdir', glob($this->base . '/*') ?: []);
            rmdir($this->base);
        }
    }

    private function generation(int $gen, float $best): GenerationCompleted
    {
        return new GenerationCompleted(
            generation: $gen,
            totalGenerations: 3,
            bestFitness: $best,
            averageFitness: $best / 2,
            allTimeBestFitness: $best,
            populationSize: 50,
            bestHiddenCount: 1,
            bestGeneCount: 2,
            improved: true,
            durationSeconds: 0.01,
            bestGenome: new Genome([Gene::of(NodeType::Input, 0, NodeType::Output, 0, 1.5)]),
            islands: [new IslandStat(0, 50, $best, $best / 2, mutationScale: 1.5)],
        );
    }

    public function testWritesStreamMetaAndBest(): void
    {
        $reporter = new JsonStreamReporter($this->store);
        $reporter->onEvent(new RunStarted('demo', EvolutionConfig::default()->name('demo'), 3, 1));
        $reporter->onEvent($this->generation(1, 1.0));
        $reporter->onEvent($this->generation(2, 2.0));
        $reporter->onEvent(new RunEnded('demo', 2, 2.0));

        $lines = file($this->store->streamPath('demo'), FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES);
        $this->assertCount(2, $lines);

        $first = json_decode($lines[0], true);
        $this->assertSame(1, $first['gen']);
        $this->assertSame(3, $first['network']['inputs']);
        $this->assertCount(1, $first['network']['genes'], 'best network is embedded for the UI');
        $this->assertEqualsWithDelta(1.5, $first['islands'][0]['mutationScale'], 1e-9);

        $meta = json_decode(file_get_contents($this->store->metaPath('demo')), true);
        $this->assertSame('demo', $meta['problem']);
        $this->assertSame(3, $meta['inputs']);

        $best = json_decode(file_get_contents($this->store->bestPath('demo')), true);
        $this->assertEqualsWithDelta(2.0, $best['bestFitness'], 1e-9);
    }

    public function testStreamCarriesAnElapsedTotalThatContinuesAcrossResume(): void
    {
        $reporter = new JsonStreamReporter($this->store);
        $reporter->onEvent(new RunStarted('demo', EvolutionConfig::default()->name('demo'), 3, 1));
        $reporter->onEvent($this->generation(1, 1.0));
        $reporter->onEvent($this->generation(2, 2.0));

        $lines = file($this->store->streamPath('demo'), FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES);
        $first = json_decode($lines[0], true);
        $second = json_decode($lines[1], true);
        $this->assertIsNumeric($first['elapsedMs']);
        $this->assertGreaterThanOrEqual($first['elapsedMs'], $second['elapsedMs'], 'elapsed only climbs within a run');

        // A resume appends to the same stream and continues the whole-run clock.
        $reporter->onEvent(new RunStarted('demo', EvolutionConfig::default()->name('demo'), 3, 1, resume: true));
        $reporter->onEvent($this->generation(3, 3.0));
        $lines = file($this->store->streamPath('demo'), FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES);
        $resumed = json_decode(end($lines), true);
        $this->assertGreaterThanOrEqual($second['elapsedMs'], $resumed['elapsedMs'], 'resume keeps counting, never resets to zero');
    }

    public function testEachRunTruncatesThePreviousStream(): void
    {
        $reporter = new JsonStreamReporter($this->store);
        $reporter->onEvent(new RunStarted('demo', EvolutionConfig::default(), 3, 1));
        $reporter->onEvent($this->generation(1, 1.0));

        // A second run with the same name starts the stream over.
        $reporter->onEvent(new RunStarted('demo', EvolutionConfig::default(), 3, 1));
        $lines = file($this->store->streamPath('demo'), FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES);
        $this->assertSame([], $lines);
    }
}
