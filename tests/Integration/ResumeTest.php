<?php

declare(strict_types=1);

namespace Rotifer\Tests\Integration;

use PHPUnit\Framework\TestCase;
use Rotifer\Evolution\World;
use Rotifer\Observe\Event\GenerationCompleted;
use Rotifer\Observe\EventDispatcher;
use Rotifer\Observe\Reporter\Reporter;
use Rotifer\Problems\XorProblem;
use Rotifer\Tests\Support\CapturingReporter;

final class ResumeTest extends TestCase
{
    /**
     * The emitted champion network must be the all-time best (the simplest organism at
     * the top score), not whoever leads the current generation. With simplicity on the
     * two diverge - this generation's live champion can be a more complex equal-scoring
     * organism. The dashboard draws the emitted network as "the champion" and force-
     * redraws it on a resume's first record, so emitting the live champion there is what
     * made a continued run show a more complex organism than the run had settled on.
     */
    public function testEmittedChampionNetworkTracksTheAllTimeBest(): void
    {
        $problem = new XorProblem();
        $config = $problem->config()->seed(5)->population(60)->generations(40)->islands(2)->simplicity(3);

        $world = new World($problem, dispatcher: (new EventDispatcher())->add($capture = new CapturingReporter()), config: $config);

        // After each generation the just-emitted event must describe the all-time best.
        $world->run(function (World $w) use ($capture): void {
            $event = end($capture->events);
            self::assertInstanceOf(GenerationCompleted::class, $event);
            $best = $w->best();
            self::assertSame($best->genome()->count(), $event->bestGeneCount, "gen {$event->generation} gene count");
            self::assertSame($best->hiddenCount(), $event->bestHiddenCount, "gen {$event->generation} hidden count");
        });
    }


    public function testRestoreContinuesWhereTheRunStopped(): void
    {
        $problem = new XorProblem();
        $config = $problem->config()->population(60)->generations(8)->islands(2);

        $first = new World($problem, config: $config);
        $first->run();
        $snapshot = $first->snapshot();

        $this->assertSame(8, $snapshot['generation']);
        $this->assertCount(2, $snapshot['islands']);

        $second = new World($problem, config: $config);
        $second->restore($snapshot);
        $second->run(); // eight more generations

        // The generation counter continues, and the best never regresses.
        $this->assertSame(16, $second->generation());
        $this->assertGreaterThanOrEqual($first->bestFitness(), $second->bestFitness());
    }

    /**
     * A resumed run must keep its adaptive-mutation pressure, not snap back to the
     * default. Without this, the first generation after a "continue" mutates with a
     * reset scale, which visibly perturbs the population's average fitness.
     */
    public function testResumeKeepsAdaptiveMutationPressure(): void
    {
        $problem = new XorProblem();
        $config = static fn (int $gens): \Rotifer\Runtime\EvolutionConfig =>
            (new XorProblem())->config()->seed(7)->population(60)->generations($gens)->adaptiveMutation(true);

        // Run 30 generations and keep the checkpoint from the last one.
        $first = new World($problem, config: $config(30));
        $snapshot = [];
        $first->run(function (World $w) use (&$snapshot): void {
            $snapshot = $w->snapshot();
        });

        // One more generation, resumed from the checkpoint.
        $resumed = new World($problem, dispatcher: ($d1 = new EventDispatcher())->add($r1 = $this->scaleRecorder()), config: $config(1));
        $resumed->restore($snapshot);
        $resumed->run();

        // The same generation reached in one continuous 31-generation run.
        $continuous = new World($problem, dispatcher: ($d2 = new EventDispatcher())->add($r2 = $this->scaleRecorder()), config: $config(31));
        $continuous->run();

        // The pressure should have moved away from the 1.0 default by now...
        $this->assertNotEqualsWithDelta(1.0, $r2->scale[31], 1e-6, 'expected adaptive mutation to have ramped');
        // ...and the resumed run's first generation should match the continuous run
        // exactly, proving the adaptive state was carried over rather than reset.
        $this->assertEqualsWithDelta($r2->scale[31], $r1->scale[31], 1e-9);
    }

    private function scaleRecorder(): Reporter
    {
        return new class implements Reporter {
            /** @var array<int, float> generation => island-0 mutation scale */
            public array $scale = [];

            public function onEvent(object $event): void
            {
                if ($event instanceof GenerationCompleted && isset($event->islands[0])) {
                    $this->scale[$event->generation] = $event->islands[0]->mutationScale;
                }
            }
        };
    }
}
