<?php

declare(strict_types=1);

namespace Rotifer\Tests\Integration;

use PHPUnit\Framework\TestCase;
use Rotifer\Evolution\ParallelWorld;
use Rotifer\Observe\EventDispatcher;
use Rotifer\Problems\XorProblem;
use Rotifer\Runtime\EvolutionConfig;

final class IslandParallelTest extends TestCase
{
    private function evolve(EvolutionConfig $config): float
    {
        $world = new ParallelWorld(new XorProblem(), $config, new EventDispatcher(), workers: 2);
        $world->run(); // creates and shuts down its own worker pool
        return $world->bestFitness();
    }

    public function testIslandParallelImprovesAndIsReproducible(): void
    {
        $config = (new XorProblem())->config()
            ->population(40)
            ->islands(2)
            ->generations(40)
            ->seed(123);

        $first = $this->evolve($config);
        $second = $this->evolve($config);

        // It actually evolves a decent solution (random XOR scores well under 2).
        $this->assertGreaterThan(2.5, $first, 'island-parallel makes real progress on XOR');
        // Per-mode determinism (option B): same seed => same parallel result.
        $this->assertEqualsWithDelta($first, $second, 1e-9, 'same seed reproduces the same parallel run');
    }

    /**
     * A parallel run's "Continue" (resume) must pick up the saved population and keep
     * evolving, not restart from generation 0. The snapshot carries every island's
     * population, so a fresh ParallelWorld restored from it continues the run.
     */
    public function testRestoreContinuesWhereTheRunStopped(): void
    {
        $config = (new XorProblem())->config()->population(40)->islands(2)->generations(8)->seed(123);

        $first = new ParallelWorld(new XorProblem(), $config, new EventDispatcher(), workers: 2);
        $first->run();
        $snapshot = $first->snapshot();

        $this->assertSame(8, $snapshot['generation']);
        $this->assertCount(2, $snapshot['islands']);
        $this->assertNotEmpty($snapshot['islands'][0], 'the island population is saved, not just the champion');

        $second = new ParallelWorld(new XorProblem(), $config, new EventDispatcher(), workers: 2);
        $second->restore($snapshot);
        $second->run(); // eight more generations, continued from the saved population

        // The generation counter continues from 8 rather than restarting, and the
        // best never regresses.
        $this->assertSame(16, $second->generation());
        $this->assertGreaterThanOrEqual($first->bestFitness(), $second->bestFitness());
    }
}
