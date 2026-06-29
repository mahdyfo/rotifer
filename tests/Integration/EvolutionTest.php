<?php

declare(strict_types=1);

namespace Rotifer\Tests\Integration;

use PHPUnit\Framework\TestCase;
use Rotifer\Evolution\World;
use Rotifer\Observe\EventDispatcher;
use Rotifer\Persistence\Codec\HexCodec;
use Rotifer\Problems\XorProblem;
use Rotifer\Tests\Support\CapturingReporter;

final class EvolutionTest extends TestCase
{
    public function testFitnessImprovesOverGenerations(): void
    {
        $problem = new XorProblem();
        $capturing = new CapturingReporter();
        $world = new World(
            $problem,
            dispatcher: (new EventDispatcher())->add($capturing),
            config: $problem->config()->population(80)->generations(30),
        );
        $world->run();

        $series = $capturing->bestFitnessSeries();
        $this->assertCount(30, $series);
        $this->assertGreaterThan($series[0], end($series), 'best fitness improves from the first generation');
    }

    public function testAllTimeBestNeverRegresses(): void
    {
        $problem = new XorProblem();
        $capturing = new CapturingReporter();
        $world = new World(
            $problem,
            dispatcher: (new EventDispatcher())->add($capturing),
            config: $problem->config()->population(80)->generations(25),
        );
        $world->run();

        $allTime = array_map(
            static fn ($e) => $e->allTimeBestFitness,
            array_filter($capturing->events, static fn ($e) => $e instanceof \Rotifer\Observe\Event\GenerationCompleted),
        );
        $sorted = $allTime;
        sort($sorted);
        $this->assertSame(array_values($sorted), array_values($allTime), 'all-time best is monotonic non-decreasing');
    }

    public function testSameSeedReplaysIdentically(): void
    {
        $config = (new XorProblem())->config()->population(60)->generations(20)->seed(2024);

        $runA = $this->seriesFor($config);
        $runB = $this->seriesFor($config);

        $this->assertSame($runA['series'], $runB['series'], 'best-fitness trajectory is identical');
        $this->assertSame($runA['best'], $runB['best'], 'champion genome is identical');
    }

    public function testDifferentSeedsDiverge(): void
    {
        $base = (new XorProblem())->config()->population(60)->generations(20);
        $this->assertNotSame(
            $this->seriesFor($base->seed(1))['best'],
            $this->seriesFor($base->seed(2))['best'],
        );
    }

    /** @return array{series: list<float>, best: string} */
    private function seriesFor(\Rotifer\Runtime\EvolutionConfig $config): array
    {
        $capturing = new CapturingReporter();
        $world = new World(
            new XorProblem(),
            dispatcher: (new EventDispatcher())->add($capturing),
            config: $config,
        );
        $best = $world->run();

        return [
            'series' => $capturing->bestFitnessSeries(),
            'best' => (new HexCodec())->encode($best->genome()),
        ];
    }
}
