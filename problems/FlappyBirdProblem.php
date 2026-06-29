<?php

declare(strict_types=1);

namespace Rotifer\Problems;

use Rotifer\Network\Activation\Sigmoid;
use Rotifer\Network\Shape;
use Rotifer\Organism\Organism;
use Rotifer\Runtime\EvolutionConfig;
use Rotifer\Runtime\Fitness\Describable;
use Rotifer\Runtime\Fitness\Predictable;
use Rotifer\Runtime\Fitness\Problem;

/**
 * A game with no training data: the network flies a bird through pipes and is
 * scored on how far it gets. The whole episode runs inside fitness() - the
 * single data row is just a trigger - so the controller drives itself tick by
 * tick. Every bird flies the same deterministic course, making fitness a fair
 * comparison. Strategy is fully emergent.
 */
final class FlappyBirdProblem implements Problem, Predictable, Describable
{
    public function description(): string
    {
        return 'A game with no training data - the network flies a bird through pipes and is scored on how far it gets.';
    }

    private const HEIGHT = 20.0;
    private const GRAVITY = 0.5;
    private const FLAP = -2.2;
    private const PIPE_GAP = 5.0;       // narrow enough that a fixed height can't pass every pipe
    private const PIPE_SPACING = 12;
    private const MAX_TICKS = 800;
    private const SOLVE_PIPES = 25;

    public function name(): string
    {
        return 'flappy_bird';
    }

    public function shape(): Shape
    {
        // bias, height, velocity, distance-to-pipe, vertical-offset-to-gap
        return new Shape(inputs: 5, outputs: 1);
    }

    public function config(): EvolutionConfig
    {
        return EvolutionConfig::default()
            ->name($this->name())
            ->population(200)
            ->generations(20)
            ->islands(2)
            ->surviveRate(0.4)
            ->elitism(0)
            ->initialHidden(3)
            ->activation(new Sigmoid())
            ->crossover(0.5)
            ->mutation(weight: 0.85, addNeuron: 0.06, addConnection: 0.12, removeNeuron: 0.02, removeConnection: 0.03)
            ->weightMutation(count: 2, adjustmentRange: 0.8, randomizeProbability: 0.12)
            ->adaptiveMutation(true)
            ->trauma(true)
            ->migration(everyGenerations: 8, topK: 2)
            ->diversityInjection(0.06)
            ->seed(1492);
    }

    /** A single trigger row; the episode is simulated in fitness(). */
    public function data(): array
    {
        return [[[1, 0, 0, 0, 0], []]];
    }

    public function fitness(Organism $organism, array $row): float
    {
        return $this->play($organism)['score'];
    }

    public function describe(Organism $best): array
    {
        $result = $this->play($best);
        return [
            'columns' => ['metric', 'value'],
            'rows' => [
                ['score', (string) round($result['score'], 2)],
                ['pipes cleared', $result['pipes'] . ' / ' . self::SOLVE_PIPES],
                ['ticks survived', (string) $result['ticks']],
                ['outcome', $result['pipes'] >= self::SOLVE_PIPES ? 'solved the course' : 'crashed'],
            ],
        ];
    }

    /**
     * Run one full game. Frames are only collected when asked (for a replay);
     * evolution never pays for them.
     *
     * @return array{score: float, pipes: int, ticks: int, frames: list<string>}
     */
    public function play(Organism $organism, bool $collectFrames = false): array
    {
        $organism->resetMemory();

        $y = self::HEIGHT / 2;
        $velocity = 0.0;
        $pipesPassed = 0;
        $frames = [];
        $gaps = $this->course();

        for ($tick = 0; $tick < self::MAX_TICKS; $tick++) {
            $pipeIndex = intdiv($tick, self::PIPE_SPACING);
            $gapCenter = $gaps[$pipeIndex % count($gaps)];
            $distanceToPipe = self::PIPE_SPACING - ($tick % self::PIPE_SPACING);

            $organism->step([
                1.0,
                $y / self::HEIGHT,
                $velocity / 10.0 + 0.5,
                $distanceToPipe / self::PIPE_SPACING,
                ($gapCenter - $y) / self::HEIGHT + 0.5,
            ]);

            if ($organism->outputs()[0] > 0.5) {
                $velocity = self::FLAP;
            }
            $velocity += self::GRAVITY;
            $y += $velocity;

            if ($collectFrames) {
                $frames[] = $this->renderFrame($y, $gapCenter);
            }

            // Crash into floor/ceiling, or into a pipe at the moment of passing it.
            if ($y < 0 || $y > self::HEIGHT) {
                break;
            }
            if ($distanceToPipe === 1) {
                if (abs($y - $gapCenter) > self::PIPE_GAP / 2) {
                    break;
                }
                $pipesPassed++;
                if ($pipesPassed >= self::SOLVE_PIPES) {
                    break;
                }
            }
        }

        $ticks = min($tick + 1, self::MAX_TICKS);
        $score = $pipesPassed * 10.0 + $ticks * 0.05;
        return ['score' => $score, 'pipes' => $pipesPassed, 'ticks' => $ticks, 'frames' => $frames];
    }

    /** @return list<float> deterministic gap centers, so every bird flies the same course */
    private function course(): array
    {
        $rng = new \Rotifer\Runtime\Rng(20240611);
        $gaps = [];
        $margin = self::PIPE_GAP / 2 + 1.0;
        for ($i = 0; $i < 64; $i++) {
            // Wide vertical spread of gap centers forces the bird to actually track them.
            $gaps[] = $rng->floatBetween($margin, self::HEIGHT - $margin);
        }
        return $gaps;
    }

    private function renderFrame(float $y, float $gapCenter): string
    {
        $rows = [];
        for ($r = 0; $r < (int) self::HEIGHT; $r++) {
            $inGap = abs($r - $gapCenter) <= self::PIPE_GAP / 2;
            $isBird = (int) round($y) === $r;
            $rows[] = ($isBird ? '◉' : ' ') . ' ' . ($inGap ? ' ' : '█');
        }
        return implode("\n", $rows);
    }
}
