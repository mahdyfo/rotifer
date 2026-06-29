<?php

declare(strict_types=1);

namespace Rotifer\Cli;

use Throwable;
use Rotifer\Evolution\ParallelWorld;
use Rotifer\Evolution\World;
use Rotifer\Observe\EventDispatcher;
use Rotifer\Observe\Reporter\ConsoleReporter;
use Rotifer\Observe\Reporter\JsonStreamReporter;
use Rotifer\Observe\Reporter\TerminalDashboard;
use Rotifer\Persistence\Codec\HexCodec;
use Rotifer\Persistence\SnapshotStore;
use Rotifer\Runtime\EvolutionConfig;
use Rotifer\Runtime\FastRuntime;
use Rotifer\Runtime\RunOptions;
use Rotifer\Runtime\Fitness\Problem;

/**
 * The `rotifer` command. Intentionally small and dependency-free: parse a verb
 * and flags, resolve a Problem, run the World, print the outcome. Flags override
 * the problem's own config so experiments need no code change.
 */
final class Console
{
    private readonly ProblemRegistry $registry;

    public function __construct(?ProblemRegistry $registry = null)
    {
        $this->registry = $registry ?? new ProblemRegistry(dirname(__DIR__, 2) . '/problems');
    }

    /** @param list<string> $argv */
    public function run(array $argv): int
    {
        $command = $argv[0] ?? 'help';

        try {
            return match ($command) {
                'run' => $this->commandRun(array_slice($argv, 1)),
                'serve' => $this->commandServe(array_slice($argv, 1)),
                'list' => $this->commandList(),
                'help', '--help', '-h' => $this->commandHelp(),
                default => $this->commandHelp(1),
            };
        } catch (Throwable $e) {
            fwrite(STDERR, 'Error: ' . $e->getMessage() . PHP_EOL);
            return 1;
        }
    }

    /** @param list<string> $args */
    private function commandRun(array $args): int
    {
        $positional = [];
        $options = [];
        foreach ($args as $arg) {
            if (str_starts_with($arg, '--')) {
                [$key, $value] = array_pad(explode('=', substr($arg, 2), 2), 2, true);
                $options[$key] = $value;
            } else {
                $positional[] = $arg;
            }
        }

        if ($positional === []) {
            fwrite(STDERR, 'Usage: rotifer run <problem> [--seed=N] [--population=N] [--generations=N] [--quiet]' . PHP_EOL);
            return 1;
        }

        $problem = $this->registry->resolve($positional[0]);
        $config = $this->applyOverrides($problem->config(), $options);
        $quiet = isset($options['quiet']);
        $web = isset($options['web']);
        $resume = isset($options['resume']);
        $store = new SnapshotStore();

        if (!$quiet) {
            foreach (FastRuntime::diagnostics()['warnings'] as $warning) {
                fwrite(STDERR, "\033[33mWarning: " . $warning . "\033[0m" . PHP_EOL);
            }
        }

        $dispatcher = new EventDispatcher();
        if (!$quiet) {
            $dispatcher->add($web ? new ConsoleReporter(false) : new TerminalDashboard());
        }
        if ($web) {
            $dispatcher->add(new JsonStreamReporter($store));
            fwrite(STDOUT, sprintf(
                "Live dashboard data → runs/%s/. In another terminal run:  php bin/rotifer serve %s%s",
                $problem->name(),
                $problem->name(),
                PHP_EOL . PHP_EOL,
            ));
        }

        // Checkpoint after every generation (not just at the end) so a run that is
        // stopped from the dashboard - or one that hit its generation cap - can be
        // continued from where it left off, and the latest champion predictions are
        // always on disk (so Stop can still show a results table, mid-run).
        $checkpoint = $web
            ? static function (World|ParallelWorld $w) use ($store, $problem): void {
                file_put_contents($store->checkpointPath($problem->name()), json_encode($w->snapshot()));
                file_put_contents($store->predictionsPath($problem->name()), json_encode($w->predictions()));
            }
            : null;

        // --parallel evolves each island in its own worker (coarse-grained): the only
        // parallelism worth having, since per-organism sharding loses to its own IPC.
        // --islands sets how many demes; --parallel just runs them at once. Engages at
        // 2+ workers (1 = serial); a bare --parallel uses one worker per island.
        $parallelOpt = $options['parallel'] ?? null;
        $useParallel = $parallelOpt !== null
            && ($parallelOpt === true || (int) $parallelOpt >= 2)
            && $this->islandParallelOk($problem, $config, $quiet);

        if ($useParallel) {
            $workers = $parallelOpt === true ? 0 : (int) $parallelOpt;
            $engine = new ParallelWorld($problem, $config, $dispatcher, $workers);
        } else {
            $engine = new World($problem, dispatcher: $dispatcher, config: $config);
        }
        // Continue a previous run from its saved checkpoint, if asked and present. Both
        // the serial World and the parallel ParallelWorld share the checkpoint format,
        // so "Continue" works in either mode.
        if ($resume) {
            $checkpointPath = $store->checkpointPath($problem->name());
            if (is_file($checkpointPath)) {
                $engine->restore(json_decode((string) file_get_contents($checkpointPath), true) ?: []);
            }
        }
        $best = $engine->run($checkpoint);

        fwrite(STDOUT, PHP_EOL . $this->summary($problem, $engine->bestFitness(), $best->genome()->count(), $best->hiddenCount()) . PHP_EOL);
        $this->printPredictions($engine->predictions());
        fwrite(STDOUT, 'best genome (hex): ' . (new HexCodec())->encode($best->genome()) . PHP_EOL);

        return 0;
    }

    /** When --parallel can't be honoured (worker can't rebuild the problem, or lifetime learning is on), note it and fall back to serial. */
    private function islandParallelOk(Problem $problem, EvolutionConfig $config, bool $quiet): bool
    {
        $constructor = (new \ReflectionClass($problem))->getConstructor();
        if ($constructor !== null && $constructor->getNumberOfRequiredParameters() > 0) {
            if (!$quiet) {
                fwrite(STDOUT, 'Note: this problem can\'t be rebuilt in a worker, so --parallel is ignored (running serially).' . PHP_EOL);
            }
            return false;
        }
        if ($config->isLifetimeLearningEnabled()) {
            if (!$quiet) {
                fwrite(STDOUT, 'Note: lifetime learning runs serially, so --parallel is ignored for this run.' . PHP_EOL);
            }
            return false;
        }
        return true;
    }

    /**
     * Fold the CLI flags into the problem's config. Every knob is mapped in one
     * place - {@see RunOptions}, which the dashboard shares - so the command line
     * and the web stay at parity.
     *
     * @param array<string, string|bool> $options
     */
    private function applyOverrides(EvolutionConfig $config, array $options): EvolutionConfig
    {
        return RunOptions::applyTo($config, $options);
    }

    /**
     * Start the one persistent dashboard server. It is run-agnostic: it lists
     * problems, starts/stops runs, and follows whichever run is active - so you
     * never need a separate port per experiment.
     *
     * @param list<string> $args
     */
    private function commandServe(array $args): int
    {
        $options = [];
        foreach ($args as $arg) {
            if (str_starts_with($arg, '--')) {
                [$key, $value] = array_pad(explode('=', substr($arg, 2), 2), 2, true);
                $options[$key] = $value;
            }
        }

        $port = isset($options['port']) ? (int) $options['port'] : 8080;
        $server = dirname(__DIR__) . '/Web/server.php';

        fwrite(STDOUT, sprintf(
            "Rotifer dashboard → \033[36mhttp://127.0.0.1:%d\033[0m%sStart runs from the control panel, or with `rotifer run <problem> --web`. Ctrl+C to stop.%s",
            $port,
            PHP_EOL,
            PHP_EOL,
        ));

        // Run the dev server but filter its very chatty per-connection logging
        // (the dashboard polls a few times a second, so "Accepted/Closing" and
        // routine 200s would flood the console). Real errors still show.
        $command = escapeshellarg(PHP_BINARY) . ' -S 127.0.0.1:' . $port . ' ' . escapeshellarg($server);
        $process = proc_open($command, [STDIN, STDOUT, ['pipe', 'w']], $pipes, dirname($server, 2));
        if (!is_resource($process)) {
            return 1;
        }
        while (($line = fgets($pipes[2])) !== false) {
            if (!$this->isServerNoise($line)) {
                fwrite(STDERR, $line);
            }
        }
        return proc_close($process);
    }

    private function isServerNoise(string $line): bool
    {
        return str_contains($line, ' Accepted')
            || str_contains($line, ' Closing')
            || (bool) preg_match('#\[200\]: (GET|POST|HEAD) /(api/|app\.js|style\.css|$)#', $line);
    }

    private function commandList(): int
    {
        $problems = $this->registry->all();
        if ($problems === []) {
            fwrite(STDOUT, 'No problems found under problems/.' . PHP_EOL);
            return 0;
        }
        fwrite(STDOUT, 'Available problems:' . PHP_EOL);
        foreach ($problems as $problem) {
            fwrite(STDOUT, sprintf('  %-20s %s' . PHP_EOL, $problem['name'], $problem['class']));
        }
        return 0;
    }

    private function commandHelp(int $code = 0): int
    {
        fwrite($code === 0 ? STDOUT : STDERR, <<<TXT
            Rotifer - biologically-inspired neuroevolution

            Usage:
              rotifer run <problem> [options]   Evolve a solution to a problem
              rotifer serve <problem> [--port]  Open the live web dashboard for a run
              rotifer list                      List available problems
              rotifer help                      Show this help

            Run options (a problem's own config() is the baseline; these override it;
            shown in [] is the framework default from EvolutionConfig::default()).
            Every option below is also settable from the dashboard's control panel.

            Core:
              --seed=N           Random seed - same seed reproduces a run exactly   [1]
              --population=N     Population size                                     [150]
              --generations=N    Generations to run (0 = unbounded)                  [50]
              --islands=N        Number of semi-isolated islands                     [1]
              --parallel[=N]     Evolve each island in its own worker process (one
                                 per island; scales when evaluation is heavy)        [off]
              --web              Stream the run to runs/<name>/ for the dashboard    [off]
              --resume           Continue from the saved checkpoint                  [off]
              --quiet            Suppress the live dashboard                         [off]

            Structure / selection:
              --survive-rate=F   Fraction of each island kept as parents            [0.5]
              --elitism=N        Top organisms copied unchanged into the next gen   [1]
              --diversity=F      Fraction of fresh random organisms injected/gen     [0.0]
              --initial-hidden=N Hidden neurons a freshly seeded organism starts     [1]
              --hidden-layers=L  Fixed layered MLP, e.g. 5,3,5 (empty = dynamic)     [dynamic]
              --simplicity=N     Sig-figs of fitness tied so simpler nets win (0=off) [3]
              --activation=NAME  sigmoid|relu|leaky_relu|tanh|threshold|gelu|softmax [sigmoid]

            Reproduction:
              --crossover=F          Chance offspring mix two parents               [0.5]
              --weight-mutation=F    Chance to nudge weights                        [0.4]
              --weight-count=N       Weights nudged per mutation                    [1]
              --weight-adjust=F      Max size of a weight nudge                     [0.5]
              --weight-randomize=F   Chance a nudge fully randomises the weight     [0.1]
              --add-neuron=F         Chance to grow a neuron                        [0.03]
              --add-connection=F     Chance to add a connection                     [0.05]
              --remove-neuron=F      Chance to prune a neuron                       [0.02]
              --remove-connection=F  Chance to prune a connection                   [0.03]

            Biology (each mechanism off by default; a bare toggle uses sensible values):
              --trauma[=0|1]            Heritable, decaying stress boost
              --trauma-intensity=F     Stress applied on hardship                   [0.4]
              --trauma-decay=F         How fast inherited stress fades              [0.5]
              --adaptive-mutation[=0|1] Mutate more when stuck, less when improving
              --adaptive-patience=N    Stalled gens before ramping up              [6]
              --adaptive-up=F          Scale-up factor when stuck                   [1.5]
              --adaptive-down=F        Scale-down factor when improving             [0.95]
              --adaptive-min=F         Lowest mutation scale                        [0.25]
              --adaptive-max=F         Highest mutation scale                       [4.0]
              --lifetime-learning[=0|1] Within-life weight tuning (runs serial)
              --lifetime-steps=N       Tuning steps per evaluation                  [5 when on]
              --lifetime-step-size=F   Size of each tuning step                     [0.3]
              --lamarckian=F           Fraction of learning written back (inherited) [0.3 when on]

            Migration (needs 2+ islands):
              --migration-every=N    Trade top organisms every N generations (0=never) [0]
              --migration-top=N      How many top organisms migrate each time        [1]

            Live dashboard (one persistent server, all runs):
              Terminal 1:  php bin/rotifer serve              then open http://localhost:8080
              Terminal 2:  php bin/rotifer run <problem> --web
            Or start runs straight from the dashboard's control panel.

            TXT);
        return $code;
    }

    /** @param array{columns: list<string>, rows: list<list<string>>} $predictions */
    private function printPredictions(array $predictions): void
    {
        $columns = $predictions['columns'] ?? [];
        $rows = $predictions['rows'] ?? [];
        if ($columns === [] || $rows === []) {
            return;
        }

        $widths = [];
        foreach ($columns as $i => $col) {
            $widths[$i] = strlen((string) $col);
        }
        foreach ($rows as $row) {
            foreach ($row as $i => $cell) {
                $widths[$i] = max($widths[$i] ?? 0, strlen((string) $cell));
            }
        }

        $render = static function (array $cells) use ($widths): string {
            $out = [];
            foreach ($cells as $i => $cell) {
                $out[] = str_pad((string) $cell, $widths[$i]);
            }
            return '  ' . implode('   ', $out);
        };

        $heading = "\033[1mChampion predictions\033[0m";
        if (isset($predictions['successRate'])) {
            $heading .= sprintf("  \033[32m%d%% success\033[0m", (int) round($predictions['successRate'] * 100));
        }
        fwrite(STDOUT, PHP_EOL . $heading . PHP_EOL);
        fwrite(STDOUT, "\033[2m" . $render($columns) . "\033[0m" . PHP_EOL);
        foreach ($rows as $row) {
            fwrite(STDOUT, $render($row) . PHP_EOL);
        }
        fwrite(STDOUT, PHP_EOL);
    }

    private function summary(Problem $problem, float $bestFitness, int $genes, int $hidden): string
    {
        return sprintf(
            'Problem "%s" finished. Best fitness %.6f | genes %d | hidden %d',
            $problem->name(),
            $bestFitness,
            $genes,
            $hidden,
        );
    }
}
