<?php

require 'vendor/autoload.php';

use Rotifer\Activations\Activation;
use Rotifer\Models\Agent;
use Rotifer\Models\World;

/**
 * Flappy Bird — gradient-free control through neuroevolution.
 *
 * Unlike the other examples (XOR, autoencoder, house prices, ...) there is NO
 * labelled "right answer" here. The network never sees a target output. It only
 * gets a score for how far the bird flew, and evolution does the rest.
 *
 * Why this needs a different shape than the supervised examples:
 *   In a supervised problem World::step() feeds each agent a fixed row of inputs.
 *   But in a game, the next input depends on what the bird just did. So instead
 *   we run the ENTIRE game simulation inside the fitness function, and hand the
 *   framework a single placeholder data row. The fitness function IS the
 *   environment.
 *
 * Network sensors (5 inputs, including bias):
 *   [0] bias (always 1)
 *   [1] bird height            (0 = top, 1 = bottom)
 *   [2] bird vertical velocity (scaled)
 *   [3] horizontal distance to the next pipe
 *   [4] vertical offset from the bird to the centre of the next gap (signed)
 * Output (1): flap if > 0.5, otherwise fall.
 *
 * Options
 *      --quiet     Hide per-generation evolution logs
 *      --no-play   Skip the final animated replay of the champion
 */

const PROBABILITY_CROSSOVER = 0.5;
const PROBABILITY_MUTATE_WEIGHT = 0.5;       // weights drift often (fine motor tuning)
const MUTATE_WEIGHT_COUNT = 2;
const PROBABILITY_MUTATE_ADD_NEURON = 0.05;  // let the brain grow its own structure
const PROBABILITY_MUTATE_REMOVE_NEURON = 0.04;
const PROBABILITY_MUTATE_ADD_GENE = 0.12;
const PROBABILITY_MUTATE_REMOVE_GENE = 0.10;
const ACTIVATION = [Activation::class, 'sigmoid'];
const SAVE_WORLD_EVERY_GENERATION = 0;
const CALCULATE_STEP_TIME = false;
const ONLY_CALCULATE_FIRST_STEP_TIME = false;

// ----------------------------------------------------------------------------
// Game configuration
// ----------------------------------------------------------------------------
const HEIGHT       = 18;   // playfield rows
const VIEW_WIDTH   = 52;   // columns shown during replay
const BIRD_X       = 12;   // bird's fixed screen column
const PIPE_START_X = 34;   // where the first pipe scrolls in from
const PIPE_SPACING = 16;   // columns between consecutive pipes
const PIPE_WIDTH   = 2;    // pipe thickness in columns
const GAP_SIZE     = 6;    // vertical opening in each pipe
const GRAVITY      = 0.40; // downward pull per tick (down = +y)
const FLAP_IMPULSE = -1.6; // upward kick when the bird flaps
const MAX_VELOCITY = 3.0;
const MAX_TICKS    = 2200; // hard cap so a perfect bird still terminates
const SOLVE_PIPES  = 40;   // passing this many pipes counts as "solved"

/**
 * Deterministic gap position for pipe $i in a given $seed.
 * Using a hash (instead of the global RNG) keeps every bird in a generation
 * facing the exact same course for a fair comparison, without disturbing the
 * genetic algorithm's own randomness.
 */
function pipeGapTop(int $seed, int $i): int
{
    $h = abs(($seed * 73856093) ^ (($i + 1) * 19349663));
    $range = max(1, HEIGHT - GAP_SIZE - 2);
    return 1 + ($h % $range);
}

/**
 * Simulate one full game for a single agent.
 *
 * @param Agent      $agent      the bird's brain
 * @param int        $seed       course seed (identical for the whole generation)
 * @param callable|null $onFrame called with the game state each tick, for replay
 * @return float                 score = pipes * 10 + ticks * 0.05
 */
function playGame(Agent $agent, int $seed, ?callable $onFrame = null): float
{
    $agent->reset();

    $birdY    = HEIGHT / 2.0;
    $velocity = 0.0;
    $scroll   = 0;          // how far the world has scrolled left
    $nextPipe = 0;          // index of the next pipe to clear
    $passed   = 0;
    $ticks    = 0;

    while ($ticks < MAX_TICKS && $passed < SOLVE_PIPES) {
        // Where is the next pipe on screen, and where is its gap?
        $pipeCol  = PIPE_START_X + $nextPipe * PIPE_SPACING - $scroll;
        $gapTop   = pipeGapTop($seed, $nextPipe);
        $gapMid   = $gapTop + GAP_SIZE / 2.0;
        $distance = max(0, $pipeCol - BIRD_X);

        // Sense -> think -> act
        $agent->step([
            1.0,
            $birdY / HEIGHT,
            $velocity / MAX_VELOCITY,
            $distance / PIPE_SPACING,
            ($gapMid - $birdY) / HEIGHT,
        ]);
        $flap = $agent->getOutputValues()[0] > 0.5;

        // Physics
        if ($flap) {
            $velocity = FLAP_IMPULSE;
        }
        $velocity = max(-MAX_VELOCITY, min(MAX_VELOCITY, $velocity + GRAVITY));
        $birdY += $velocity;
        $scroll++;
        $ticks++;

        // Recompute the pipe position after scrolling
        $pipeCol = PIPE_START_X + $nextPipe * PIPE_SPACING - $scroll;

        // Did we just clear the pipe completely?
        if ($pipeCol + PIPE_WIDTH - 1 < BIRD_X) {
            $passed++;
            $nextPipe++;
            $pipeCol = PIPE_START_X + $nextPipe * PIPE_SPACING - $scroll;
            $gapTop  = pipeGapTop($seed, $nextPipe);
        }

        $dead = false;
        // Hit the floor or ceiling?
        if ($birdY < 0 || $birdY >= HEIGHT) {
            $dead = true;
        }
        // Inside a pipe's columns but outside its gap?
        $row = (int) round($birdY);
        if (!$dead && BIRD_X >= $pipeCol && BIRD_X <= $pipeCol + PIPE_WIDTH - 1) {
            if ($row < $gapTop || $row >= $gapTop + GAP_SIZE) {
                $dead = true;
            }
        }

        if ($onFrame !== null) {
            $onFrame($birdY, $velocity, $scroll, $nextPipe, $passed, $ticks, $seed, $dead);
        }

        if ($dead) {
            break;
        }
    }

    return $passed * 10 + $ticks * 0.05;
}

/**
 * Render a single frame of the game to the terminal.
 */
function renderFrame(float $birdY, int $scroll, int $passed, int $generation, int $score, bool $dead, int $seed): void
{
    $grid = array_fill(0, HEIGHT, array_fill(0, VIEW_WIDTH, ' '));

    // Draw every pipe currently on screen
    for ($i = 0; $i < 200; $i++) {
        $col = PIPE_START_X + $i * PIPE_SPACING - $scroll;
        if ($col + PIPE_WIDTH - 1 < 0 || $col >= VIEW_WIDTH) {
            continue;
        }
        $gapTop = pipeGapTop($seed, $i);
        for ($row = 0; $row < HEIGHT; $row++) {
            if ($row >= $gapTop && $row < $gapTop + GAP_SIZE) {
                continue; // the opening
            }
            for ($w = 0; $w < PIPE_WIDTH; $w++) {
                $c = $col + $w;
                if ($c >= 0 && $c < VIEW_WIDTH) {
                    $grid[$row][$c] = '#';
                }
            }
        }
    }

    // Draw the bird
    $brow = max(0, min(HEIGHT - 1, (int) round($birdY)));
    $grid[$brow][BIRD_X] = $dead ? 'X' : '@';

    // Compose the screen
    $out = "\033[H\033[2J"; // move cursor home + clear
    $out .= sprintf(
        "  Flappy Rotifer  |  Gen %d champion  |  Pipes: %d  |  Score: %d%s\n",
        $generation,
        $passed,
        $score,
        $dead ? '   <CRASH>' : ''
    );
    $out .= '  +' . str_repeat('-', VIEW_WIDTH) . "+\n";
    foreach ($grid as $row) {
        $out .= '  |' . implode('', $row) . "|\n";
    }
    $out .= '  +' . str_repeat('-', VIEW_WIDTH) . "+\n";
    echo $out;
    flush();
}

// ----------------------------------------------------------------------------
// Evolve
// ----------------------------------------------------------------------------
$population  = 200;
$generations = 60;

// A single placeholder row: the game itself happens inside the fitness function.
// The 5 zeros just match the network's input count for the framework's pre-step.
$data = [
    [[0, 0, 0, 0, 0], []],
];

$fitnessFunction = function (Agent $agent, $dataRow, $otherAgents, World $world): float {
    // Same course for every bird in this generation; new course each generation.
    return playGame($agent, $world->getGeneration());
};

// Stop early once a bird masters the game.
$stopFunction = function (World $world): bool {
    return $world->getBestAgent()->getFitness() >= SOLVE_PIPES * 10;
};

echo "Evolving flappy birds (no backprop, no labels — just survival)...\n\n";

$world = new World('flappy');
$world->createAgents($population, 5, 1); // 5 inputs, 1 output, dynamic architecture
$world->step($fitnessFunction, $data, $generations, 0.2, 0, $stopFunction);

$best = $world->getBestAgent();
$pipes = (int) floor($best->getFitness() / 10);

echo "\n";
echo "Done. Champion cleared ~{$pipes} pipes "
    . "(fitness " . number_format($best->getFitness(), 1) . ").\n";
echo "Evolved brain: "
    . count($best->getGenomeArray()) . " connections, "
    . count($best->getNeuronsByType(\Rotifer\Models\Neuron::TYPE_HIDDEN)) . " hidden neurons.\n";

// ----------------------------------------------------------------------------
// Replay the champion with live animation
// ----------------------------------------------------------------------------
if (in_array('--no-play', $_SERVER['argv'] ?? [], true)) {
    return;
}

echo "\nReplaying the champion on a fresh course in 5s...\n";
sleep(5);

$replaySeed = 1234; // a course the champion has never seen
$generation = $world->getGeneration();

playGame($best, $replaySeed, function (
    $birdY, $velocity, $scroll, $nextPipe, $passed, $ticks, $seed, $dead
) use ($generation) {
    renderFrame($birdY, $scroll, $passed, $generation, $passed * 10 + (int) ($ticks * 0.05), $dead, $seed);
    usleep(55_000); // ~18 fps
});

echo "\n  Thanks for watching. Run again to evolve a new champion!\n";
