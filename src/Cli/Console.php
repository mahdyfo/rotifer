<?php

declare(strict_types=1);

namespace Rotifer\Cli;

use Throwable;
use Rotifer\Evolution\ParallelWorld;
use Rotifer\Evolution\World;
use Rotifer\Network\Activation\ActivationFactory;
use Rotifer\Observe\EventDispatcher;
use Rotifer\Observe\Reporter\ConsoleReporter;
use Rotifer\Observe\Reporter\JsonStreamReporter;
use Rotifer\Observe\Reporter\TerminalDashboard;
use Rotifer\Persistence\Codec\HexCodec;
use Rotifer\Persistence\SnapshotStore;
use Rotifer\Runtime\EvolutionConfig;
use Rotifer\Runtime\FastRuntime;
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

    /** @param array<string, string|bool> $options */
    private function applyOverrides(EvolutionConfig $config, array $options): EvolutionConfig
    {
        if (isset($options['seed'])) {
            $config = $config->seed((int) $options['seed']);
        }
        if (isset($options['population'])) {
            $config = $config->population((int) $options['population']);
        }
        if (isset($options['generations'])) {
            $config = $config->generations((int) $options['generations']);
        }
        if (isset($options['islands'])) {
            $config = $config->islands((int) $options['islands']);
        }

        // Biology toggles (so the dashboard / CLI can flip mechanisms on or off).
        if (array_key_exists('trauma', $options)) {
            $config = $config->trauma($this->boolOpt($options['trauma']));
        }
        if (array_key_exists('adaptive-mutation', $options)) {
            $config = $config->adaptiveMutation($this->boolOpt($options['adaptive-mutation']));
        }
        if (array_key_exists('lifetime-learning', $options)) {
            $config = $this->boolOpt($options['lifetime-learning'])
                ? $config->lifetimeLearning(steps: 5, lamarckian: 0.3)
                : $config->lifetimeLearning(steps: 0, enabled: false);
        }

        // Advanced overrides: activation, reproduction probabilities, structure,
        // and migration. Each only applies when the flag is present.
        if (isset($options['activation'])) {
            $config = $config->activation(ActivationFactory::fromName((string) $options['activation']));
        }
        if (isset($options['crossover'])) {
            $config = $config->crossover((float) $options['crossover']);
        }
        $config = $config->mutation(
            weight: isset($options['weight-mutation']) ? (float) $options['weight-mutation'] : null,
            addNeuron: isset($options['add-neuron']) ? (float) $options['add-neuron'] : null,
            addConnection: isset($options['add-connection']) ? (float) $options['add-connection'] : null,
            removeNeuron: isset($options['remove-neuron']) ? (float) $options['remove-neuron'] : null,
            removeConnection: isset($options['remove-connection']) ? (float) $options['remove-connection'] : null,
        );
        if (isset($options['survive-rate'])) {
            $config = $config->surviveRate((float) $options['survive-rate']);
        }
        if (isset($options['elitism'])) {
            $config = $config->elitism((int) $options['elitism']);
        }
        if (isset($options['initial-hidden'])) {
            $config = $config->initialHidden((int) $options['initial-hidden']);
        }
        if (isset($options['simplicity'])) {
            $config = $config->simplicity((int) $options['simplicity']);
        }
        // Topology: "5,3,5" pins a fixed layered network; an empty value (or "dynamic")
        // forces the evolving topology, overriding whatever the problem configured.
        if (isset($options['hidden-layers']) && is_string($options['hidden-layers'])) {
            $sizes = array_values(array_filter(
                array_map('intval', explode(',', $options['hidden-layers'])),
                static fn (int $n) => $n > 0,
            ));
            $config = $config->hiddenLayers($sizes);
        }
        if (isset($options['diversity'])) {
            $config = $config->diversityInjection((float) $options['diversity']);
        }
        if (isset($options['migration-every'])) {
            $config = $config->migration((int) $options['migration-every'], (int) ($options['migration-top'] ?? 1));
        }
        return $config;
    }

    private function boolOpt(string|bool $value): bool
    {
        if ($value === true) {
            return true; // bare flag, e.g. --trauma
        }
        return filter_var($value, FILTER_VALIDATE_BOOLEAN);
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
            shown in [] is the framework default from EvolutionConfig::default()):
              --seed=N           Random seed - same seed reproduces a run exactly   [1]
              --population=N     Population size                                     [150]
              --generations=N    Generations to run (0 = unbounded)                  [50]
              --islands=N        Number of semi-isolated islands                     [1]
              --parallel[=N]     Evolve each island in its own worker process (one
                                 per island; scales when evaluation is heavy)        [off]
              --web              Stream the run to runs/<name>/ for the dashboard    [off]
              --quiet            Suppress the live dashboard                         [off]

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
