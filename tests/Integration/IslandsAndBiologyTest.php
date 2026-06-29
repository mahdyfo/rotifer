<?php

declare(strict_types=1);

namespace Rotifer\Tests\Integration;

use PHPUnit\Framework\TestCase;
use Rotifer\Evolution\World;
use Rotifer\Persistence\Codec\HexCodec;
use Rotifer\Problems\XorProblem;
use Rotifer\Runtime\EvolutionConfig;

final class IslandsAndBiologyTest extends TestCase
{
    public function testIslandsPartitionTheConfiguredPopulation(): void
    {
        $config = (new XorProblem())->config()->population(84)->islands(4)->generations(3);
        $world = new World(new XorProblem(), config: $config);
        $world->run();

        $this->assertCount(4, $world->islands());
        $this->assertCount(84, $world->population(), 'island sizes sum to the configured population');
    }

    public function testMultiIslandRunsAreReproducible(): void
    {
        $config = (new XorProblem())->config()->population(90)->islands(3)->generations(15)->seed(555);
        $this->assertSame($this->bestHex($config), $this->bestHex($config));
    }

    public function testFullBiologyStackRunsAndImproves(): void
    {
        $config = (new XorProblem())->config()
            ->population(120)
            ->islands(3)
            ->generations(45)
            ->trauma(true)
            ->adaptiveMutation(true)
            ->lifetimeLearning(steps: 3, lamarckian: 0.3)
            ->migration(everyGenerations: 5, topK: 2)
            ->seed(123);

        $world = new World(new XorProblem(), config: $config);
        $world->run();

        // Every biological path is exercised; the run should make real progress.
        $this->assertGreaterThan(3.0, $world->bestFitness());
    }

    private function bestHex(EvolutionConfig $config): string
    {
        $world = new World(new XorProblem(), config: $config);
        return (new HexCodec())->encode($world->run()->genome());
    }
}
