<?php

declare(strict_types=1);

namespace Rotifer\Evolution;

use Amp\Parallel\Worker\ContextWorkerPool;
use Amp\Parallel\Worker\WorkerPool;
use Rotifer\Evolution\Migration\MigrationPolicy;
use Rotifer\Genome\Genome;
use Rotifer\Network\LayerPlan;
use Rotifer\Network\NetworkSpec;
use Rotifer\Observe\Event\GenerationCompleted;
use Rotifer\Observe\Event\IslandStat;
use Rotifer\Observe\Event\RunEnded;
use Rotifer\Observe\Event\RunStarted;
use Rotifer\Observe\EventDispatcher;
use Rotifer\Organism\Organism;
use Rotifer\Runtime\EvolutionConfig;
use Rotifer\Runtime\FastRuntime;
use Rotifer\Runtime\Fitness\Predictor;
use Rotifer\Runtime\Fitness\Problem;
use Rotifer\Runtime\Parallel\IslandEpochTask;

use function Amp\Future\await;

/**
 * The coarse-grained, island-parallel counterpart to {@see World}.
 *
 * Each island lives in its own worker process and evolves a whole migration epoch
 * (K = migration interval generations) there before reporting back; the main
 * process only migrates between islands, tracks the global champion, and emits one
 * {@see GenerationCompleted} per epoch. No genome crosses the boundary mid-epoch,
 * so this scales near-linearly with worker count when evaluation is the bottleneck
 * - unlike the per-generation sharding in {@see World}, whose IPC dominates.
 *
 * Determinism is per-mode (option B): the same seed replays the same parallel run,
 * but a parallel run is not byte-identical to a serial one. Serial stays the
 * default; this is opt-in via the CLI's --parallel.
 */
final class ParallelWorld
{
    private readonly NetworkSpec $spec;
    private readonly MigrationPolicy $migration;
    /** @var callable(Organism, Organism): int */
    private $fitter;

    /** @var list<int> island sizes */
    private array $sizes;
    /** @var list<list<array{genome: list<array<string,int|float>>, markers: array<string,float>}>> per-island population snapshot */
    private array $populations = [];
    /** @var list<array<string,mixed>> per-island state snapshot */
    private array $states = [];

    private int $generation = 0;
    private bool $resuming = false;
    private ?Genome $bestGenome = null;
    private float $bestFitness = -INF;
    private int $bestHidden = 0;
    private int $bestGenes = 0;
    private ?float $bestMatchRate = null;
    /** @var array{columns: list<string>, rows: list<list<string>>} */
    private array $predictions = ['columns' => [], 'rows' => []];

    public function __construct(
        private readonly Problem $problem,
        private readonly EvolutionConfig $config,
        private readonly EventDispatcher $dispatcher,
        private readonly int $workers = 0,
    ) {
        $layers = $config->getHiddenLayers();
        $this->spec = new NetworkSpec(
            $problem->shape(),
            $config->hasMemory(),
            $config->getActivation(),
            $layers === [] ? null : new LayerPlan($layers),
        );
        $this->migration = new MigrationPolicy($config->getMigrationEveryGenerations(), $config->getMigrationTopK());
        $this->fitter = Organism::ranker($config->getSimplicity());
        $this->sizes = $this->islandSizes($config->getIslands(), $config->getPopulation());
        $this->populations = array_fill(0, count($this->sizes), []);
        $this->states = array_fill(0, count($this->sizes), []);
    }

    /** @param ?callable(self): void $afterGeneration run after each epoch (e.g. to checkpoint) */
    public function run(?callable $afterGeneration = null, ?WorkerPool $pool = null): Organism
    {
        $this->dispatcher->dispatch(new RunStarted(
            $this->problem->name(),
            $this->config,
            $this->spec->inputs(),
            $this->spec->outputs(),
            $this->resuming,
        ));

        // Give the workers OPcache JIT too - but only when this process is itself
        // accelerated. Otherwise (e.g. under Xdebug coverage in tests) it would just
        // warn that JIT is incompatible with Xdebug and achieve nothing.
        if (FastRuntime::diagnostics()['jit']) {
            putenv('PHP_INI_SCAN_DIR=' . FastRuntime::workerIniDir());
        }
        $islandCount = count($this->sizes);
        $pool ??= new ContextWorkerPool(max(1, $this->workers > 0 ? $this->workers : $islandCount));

        try {
            $epochLength = $this->epochLength();
            $totalGenerations = $this->config->getGenerations();
            $target = $totalGenerations === 0 ? 0 : $this->generation + $totalGenerations;
            $epoch = 0;

            while ($target === 0 || $this->generation < $target) {
                $remaining = $target === 0 ? $epochLength : min($epochLength, $target - $this->generation);
                $start = microtime(true);
                $results = $this->runEpoch($pool, $epoch, $remaining);
                $this->generation += $remaining;
                $this->migrate($results);
                $improved = $this->absorbChampion($results);
                $this->dispatcher->dispatch($this->epochEvent($results, $target, $improved, microtime(true) - $start));
                if ($afterGeneration !== null) {
                    $afterGeneration($this);
                }
                $epoch++;
            }
        } finally {
            $pool->shutdown();
        }

        $best = $this->best();
        $this->predictions = Predictor::describe($this->problem, $best);
        $this->dispatcher->dispatch(new RunEnded($this->problem->name(), $this->generation, $this->bestFitness(), $this->predictions));
        return $best;
    }

    /**
     * Dispatch one epoch: every island runs $generations generations in a worker.
     * @return list<array<string,mixed>> results indexed by island
     */
    private function runEpoch(WorkerPool $pool, int $epoch, int $generations): array
    {
        $executions = [];
        foreach ($this->sizes as $i => $size) {
            $executions[$i] = $pool->submit(new IslandEpochTask(
                $this->problem::class,
                $this->spec,
                $this->config,
                $i,
                $size,
                $this->config->getSeed(),
                $epoch,
                $this->generation,
                $generations,
                $this->populations[$i],
                $this->states[$i],
                $this->migration->topK(),
            ));
        }
        $results = await(array_map(static fn ($e) => $e->getFuture(), $executions));

        // Carry each island's resulting population + state into the next epoch.
        foreach ($results as $i => $result) {
            $this->populations[$i] = $result['population'];
            $this->states[$i] = $result['state'];
        }
        return $results;
    }

    /** @param list<array<string,mixed>> $results */
    private function migrate(array $results): void
    {
        $count = count($this->sizes);
        if ($count < 2) {
            return;
        }
        foreach ($results as $sourceIndex => $result) {
            $destIndex = $this->migration->destinationOf($sourceIndex, $count);
            if ($destIndex === $sourceIndex) {
                continue;
            }
            $migrants = $result['migrants'] ?? [];
            $destPop = $this->populations[$destIndex];
            foreach ($migrants as $k => $genome) {
                $slot = count($destPop) - 1 - $k;
                if ($slot <= 0) {
                    break;
                }
                $destPop[$slot] = ['genome' => $genome, 'markers' => []];
            }
            $this->populations[$destIndex] = $destPop;
        }
    }

    /**
     * Fold the epoch's island bests into the global champion (highest fitness, or an
     * equal-fitness simpler network). Returns whether it improved.
     *
     * @param list<array<string,mixed>> $results
     */
    private function absorbChampion(array $results): bool
    {
        $improved = false;
        foreach ($results as $result) {
            $fitness = (float) $result['bestFitness'];
            $genomeArray = $result['bestGenome'] ?? [];
            if ($genomeArray === []) {
                continue;
            }
            $candidate = new Organism(Genome::fromArray($genomeArray), $this->spec, null);
            $candidate->setFitness($fitness);
            if ($this->bestGenome === null || (($this->fitter)($candidate, $this->best()) < 0 && $fitness >= $this->bestFitness)) {
                $this->bestGenome = $candidate->genome();
                $this->bestFitness = $fitness;
                $this->bestHidden = (int) $result['hidden'];
                $this->bestGenes = (int) $result['genes'];
                $improved = true;
            }
        }
        if (($improved || $this->predictions['rows'] === []) && $this->bestGenome !== null) {
            $this->predictions = Predictor::describe($this->problem, $this->best());
            $rate = $this->predictions['successRate'] ?? null;
            if ($rate !== null) {
                $this->bestMatchRate = $this->bestMatchRate === null ? $rate : max($this->bestMatchRate, $rate);
            }
        }
        return $improved;
    }

    /** @param list<array<string,mixed>> $results */
    private function epochEvent(array $results, int $total, bool $improved, float $duration): GenerationCompleted
    {
        $islands = [];
        $size = 0;
        $fitnessSum = 0.0;
        $epochBest = -INF;
        foreach ($results as $i => $result) {
            $stat = $result['stat'];
            $islands[] = new IslandStat(
                index: $i,
                size: (int) $stat['size'],
                bestFitness: (float) $stat['best'],
                averageFitness: (float) $stat['avg'],
                mutationScale: (float) $stat['mutationScale'],
                traumaLevel: (float) $stat['trauma'],
            );
            $size += (int) $stat['size'];
            $fitnessSum += (float) $stat['avg'] * (int) $stat['size'];
            $epochBest = max($epochBest, (float) $stat['best']);
        }

        return new GenerationCompleted(
            generation: $this->generation,
            totalGenerations: $total,
            bestFitness: $epochBest === -INF ? 0.0 : $epochBest,
            averageFitness: $size > 0 ? $fitnessSum / $size : 0.0,
            allTimeBestFitness: $this->bestFitness(),
            populationSize: $size,
            bestHiddenCount: $this->bestHidden,
            bestGeneCount: $this->bestGenes,
            improved: $improved,
            durationSeconds: $duration,
            bestGenome: $this->bestGenome ?? new Genome(),
            islands: $islands,
            matchRate: $this->bestMatchRate,
        );
    }

    /** @return array{columns: list<string>, rows: list<list<string>>} */
    public function predictions(): array
    {
        return $this->predictions;
    }

    public function best(): Organism
    {
        $genome = $this->bestGenome ?? new Genome();
        return (new Organism($genome, $this->spec))->setFitness($this->bestFitness());
    }

    public function bestFitness(): float
    {
        return $this->bestFitness === -INF ? 0.0 : $this->bestFitness;
    }

    public function generation(): int
    {
        return $this->generation;
    }

    /**
     * The whole living world as plain data, so a parallel run can be stopped and
     * continued later: generation count, global best, and every island's population
     * (genomes + markers) and adaptive/trauma state. The keys match {@see World::snapshot()}
     * so a checkpoint is shared between the serial and parallel engines.
     *
     * @return array<string, mixed>
     */
    public function snapshot(): array
    {
        return [
            'generation' => $this->generation,
            'bestFitness' => $this->bestFitness(),
            'bestGenome' => $this->bestGenome?->toArray() ?? [],
            'bestHidden' => $this->bestHidden,
            'bestGenes' => $this->bestGenes,
            'islands' => $this->populations,
            'islandState' => $this->states,
        ];
    }

    /**
     * Reload a {@see snapshot()} so {@see run()} continues from where it stopped: the
     * generation counter resumes, the global champion is restored, and each island's
     * saved population + state are seeded back into the next epoch.
     *
     * @param array<string, mixed> $snapshot
     */
    public function restore(array $snapshot): void
    {
        $this->resuming = true;
        $this->generation = (int) ($snapshot['generation'] ?? 0);
        $this->bestFitness = (float) ($snapshot['bestFitness'] ?? -INF);
        $bestGenome = $snapshot['bestGenome'] ?? [];
        $this->bestGenome = $bestGenome === [] ? null : Genome::fromArray($bestGenome);
        $this->bestHidden = (int) ($snapshot['bestHidden'] ?? 0);
        $this->bestGenes = (int) ($snapshot['bestGenes'] ?? 0);

        $savedPops = $snapshot['islands'] ?? [];
        $savedState = $snapshot['islandState'] ?? [];
        foreach (array_keys($this->sizes) as $i) {
            if (isset($savedPops[$i]) && is_array($savedPops[$i]) && $savedPops[$i] !== []) {
                $this->populations[$i] = $savedPops[$i];
            }
            if (isset($savedState[$i]) && is_array($savedState[$i])) {
                $this->states[$i] = $savedState[$i];
            }
        }
    }

    /** The number of generations one worker runs before reporting back (the migration interval). */
    private function epochLength(): int
    {
        $interval = $this->config->getMigrationEveryGenerations();
        if ($interval > 0) {
            return $interval;
        }
        // No migration configured: pick a sensible chunk so events still flow.
        $total = $this->config->getGenerations();
        return $total > 0 ? max(1, (int) ceil($total / 10)) : 25;
    }

    /** @return list<int> island sizes summing to the configured population */
    private function islandSizes(int $islandCount, int $population): array
    {
        $islandCount = max(1, $islandCount);
        $base = max(2, intdiv($population, $islandCount));
        $sizes = array_fill(0, $islandCount, $base);
        $remainder = $population - $base * $islandCount;
        for ($i = 0; $i < $remainder; $i++) {
            $sizes[$i]++;
        }
        return $sizes;
    }
}
