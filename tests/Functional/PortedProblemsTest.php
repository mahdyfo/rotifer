<?php

declare(strict_types=1);

namespace Rotifer\Tests\Functional;

use PHPUnit\Framework\TestCase;
use Rotifer\Evolution\World;
use Rotifer\Observe\EventDispatcher;
use Rotifer\Problems\AutoEncoderProblem;
use Rotifer\Problems\FlappyBirdProblem;
use Rotifer\Problems\HousePriceProblem;
use Rotifer\Problems\PhoneRecallProblem;
use Rotifer\Problems\WeatherForecastProblem;
use Rotifer\Runtime\Fitness\Problem;
use Rotifer\Tests\Support\CapturingReporter;

final class PortedProblemsTest extends TestCase
{
    /** Run a problem briefly and return its best-fitness-per-generation series. */
    private function series(Problem $problem, int $population, int $generations): array
    {
        $capturing = new CapturingReporter();
        $world = new World(
            $problem,
            dispatcher: (new EventDispatcher())->add($capturing),
            config: $problem->config()->population($population)->generations($generations),
        );
        $world->run();
        return $capturing->bestFitnessSeries();
    }

    public function testRegressionImproves(): void
    {
        $series = $this->series(new HousePriceProblem(), 60, 25);
        $this->assertGreaterThan($series[0], end($series), 'house-price regression improves');
    }

    public function testClassificationImproves(): void
    {
        $series = $this->series(new WeatherForecastProblem(), 80, 25);
        $this->assertGreaterThan($series[0], end($series), 'weather classification improves');
    }

    public function testAutoEncoderImproves(): void
    {
        $series = $this->series(new AutoEncoderProblem(), 80, 30);
        $this->assertGreaterThan($series[0], end($series), 'autoencoder reconstruction improves');
    }

    public function testPhoneRecallImprovesUsingMemory(): void
    {
        // Constant input every step, so the only way to do better is to evolve
        // internal memory that drives the digit sequence.
        $series = $this->series(new PhoneRecallProblem(), 120, 30);
        $this->assertGreaterThan($series[0], max($series), 'phone recall improves using memory');
    }

    public function testFlappyBirdLearnsToFlyFurther(): void
    {
        $series = $this->series(new FlappyBirdProblem(), 100, 30);
        // Gen-champion fitness fluctuates, so compare the best ever reached.
        $best = max($series);
        $this->assertGreaterThan($series[0], $best, 'flappy bird improves over the first generation');
        $this->assertGreaterThan(15.0, $best, 'best controller clears multiple pipes');
    }
}
