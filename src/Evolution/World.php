<?php

declare(strict_types=1);

namespace Rotifer\Evolution;

use Rotifer\Evolution\Adaptation\AdaptiveMutation;
use Rotifer\Evolution\Epigenetics\TraumaPolicy;
use Rotifer\Evolution\Learning\LifetimeLearner;
use Rotifer\Evolution\Migration\MigrationPolicy;
use Rotifer\Evolution\Reproduction\Reproducer;
use Rotifer\Evolution\Selection\TournamentSelection;
use Rotifer\Genome\Genome;
use Rotifer\Network\LayerPlan;
use Rotifer\Network\NetworkSpec;
use Rotifer\Observe\Event\GenerationCompleted;
use Rotifer\Observe\Event\RunEnded;
use Rotifer\Observe\Event\RunStarted;
use Rotifer\Observe\EventDispatcher;
use Rotifer\Organism\Epigenome;
use Rotifer\Organism\Organism;
use Rotifer\Runtime\EvolutionConfig;
use Rotifer\Runtime\Fitness\ClosableEvaluator;
use Rotifer\Runtime\Fitness\FitnessEvaluator;
use Rotifer\Runtime\Fitness\Predictor;
use Rotifer\Runtime\Fitness\Problem;
use Rotifer\Runtime\Fitness\SerialEvaluator;
use Rotifer\Runtime\Fitness\WindowSelector;
use Rotifer\Runtime\Rng;

/**
 * Coordinates a whole world of islands toward a solution.
 *
 * Each generation it evaluates every island, lets each breed its next cohort
 * (applying whatever biology is enabled), migrates organisms between islands on
 * schedule, tracks the global champion, and emits one event describing it all.
 * Output and persistence live in reporters; determinism lives in one seeded Rng
 * tree (the master seeds each island's stream), so a run replays exactly.
 */
final class World
{
    private readonly EvolutionConfig $config;
    private readonly NetworkSpec $spec;
    private readonly FitnessEvaluator $evaluator;
    private readonly EventDispatcher $dispatcher;
    private readonly MigrationPolicy $migration;
    private readonly IdSequence $ids;
    private readonly ?WindowSelector $windowSelector;
    private readonly int $scorableRows;

    /** @var list<Island> */
    private array $islands = [];
    private int $generation = 0;
    private bool $resuming = false;

    private ?Genome $bestGenome = null;
    private float $bestFitness = -INF;
    private ?float $bestMatchRate = null;
    /** @var callable(Organism, Organism): int */
    private $fitter;
    /** @var array{columns: list<string>, rows: list<list<string>>} */
    private array $predictions = ['columns' => [], 'rows' => []];

    public function __construct(
        private readonly Problem $problem,
        ?FitnessEvaluator $evaluator = null,
        ?EventDispatcher $dispatcher = null,
        ?EvolutionConfig $config = null,
    ) {
        $this->config = $config ?? $problem->config();
        $layers = $this->config->getHiddenLayers();
        $this->spec = new NetworkSpec(
            $problem->shape(),
            $this->config->hasMemory(),
            $this->config->getActivation(),
            $layers === [] ? null : new LayerPlan($layers),
        );
        $master = new Rng($this->config->getSeed());
        $this->fitter = Organism::ranker($this->config->getSimplicity());
        $this->evaluator = $evaluator ?? $this->defaultEvaluator($master);
        $this->dispatcher = $dispatcher ?? new EventDispatcher();
        $this->migration = new MigrationPolicy($this->config->getMigrationEveryGenerations(), $this->config->getMigrationTopK());
        $this->ids = new IdSequence();
        $this->windowSelector = WindowSelector::fromConfig($this->config);
        $this->scorableRows = $this->windowSelector === null ? 0 : WindowSelector::countScorable($problem->data());

        $this->buildIslands($master);
    }

    /** @param ?callable(self): void $afterGeneration run after each generation (e.g. to checkpoint) */
    public function run(?callable $afterGeneration = null): Organism
    {
        $this->dispatcher->dispatch(new RunStarted(
            $this->problem->name(),
            $this->config,
            $this->spec->inputs(),
            $this->spec->outputs(),
            $this->resuming,
        ));

        try {
            // When resuming, keep evolving from the restored generation count.
            $additional = $this->config->getGenerations();
            $startGen = $this->generation;
            $target = $additional === 0 ? 0 : $startGen + $additional;
            $gen = $startGen;
            while ($target === 0 || $gen < $target) {
                $gen++;
                $this->advance($gen, $target);
                if ($afterGeneration !== null) {
                    $afterGeneration($this);
                }
            }
        } finally {
            // Release any worker pool while the event loop is still alive.
            if ($this->evaluator instanceof ClosableEvaluator) {
                $this->evaluator->close();
            }
        }

        $best = $this->best();
        $this->predictions = Predictor::describe($this->problem, $best);
        $this->dispatcher->dispatch(new RunEnded($this->problem->name(), $this->generation, $this->bestFitness(), $this->predictions));
        return $best;
    }

    /** @return array{columns: list<string>, rows: list<list<string>>} */
    public function predictions(): array
    {
        return $this->predictions;
    }

    public function advance(int $gen, int $totalGenerations = 0): void
    {
        $start = microtime(true);
        $this->generation = $gen;

        // Evaluate every island's organisms in one batch. Scoring is pure and
        // independent, and organisms are mutated in place by reference, so folding
        // all islands into a single call lets the parallel evaluator shard the whole
        // population across its workers at once (one IPC round, larger chunks, busier
        // cores) instead of one undersized round per island.
        $window = $this->windowSelector?->forGeneration($gen, $this->scorableRows);
        $this->evaluator->evaluate($this->population(), $this->problem, $window);

        $emigrants = $this->migration->isDue($gen) ? $this->collectEmigrants() : null;

        foreach ($this->islands as $island) {
            $island->reproduce($gen);
        }

        if ($emigrants !== null) {
            $this->distributeEmigrants($emigrants);
        }

        $champion = $this->globalChampion();
        // A champion becomes the new all-time best when it ranks better (higher score,
        // or an equal score with a simpler network) AND its raw score is not lower
        // than the current best's. The score guard keeps the all-time best monotonic
        // (it never dips), while still letting an equal-or-better but simpler network
        // take over - so the saved best and the dashboard's champion network keep
        // improving and shedding complexity even after the score plateaus.
        $previousBest = $this->bestGenome === null ? null : $this->best();
        $improved = $previousBest === null
            || (($this->fitter)($champion, $previousBest) < 0 && $champion->fitness() >= $previousBest->fitness());
        if ($improved) {
            $this->bestFitness = $champion->fitness();
            $this->bestGenome = $champion->genome();
        }
        // Refresh the champion's predictions on improvement (and once at the start)
        // the match rate is clamped to its running max so it only ever climbs - score
        // and closeness can diverge, so a higher-fitness champion may match slightly
        // worse, but we never show the match rate dipping
        if (($improved || $this->predictions['rows'] === []) && $this->bestGenome !== null) {
            $this->predictions = Predictor::describe($this->problem, $this->best());
            $rate = $this->predictions['successRate'] ?? null;
            if ($rate !== null) {
                $this->bestMatchRate = $this->bestMatchRate === null ? $rate : max($this->bestMatchRate, $rate);
            }
        }

        $this->dispatcher->dispatch($this->generationEvent($gen, $totalGenerations, $champion, $improved, microtime(true) - $start));
    }

    private function generationEvent(int $gen, int $total, Organism $champion, bool $improved, float $duration): GenerationCompleted
    {
        $stats = array_map(static fn (Island $i) => $i->stat(), $this->islands);
        $size = 0;
        $fitnessSum = 0.0;
        foreach ($stats as $stat) {
            $size += $stat->size;
            $fitnessSum += $stat->averageFitness * $stat->size;
        }

        // The champion network and its hidden/gene counts track the all-time best (the
        // simplest organism at the top score), not whoever leads the current generation.
        // The two coincide whenever the best improves - which is the only time the
        // dashboard redraws the network during a live run - but they can diverge when the
        // view is force-drawn without an improvement (a resume, or the first record after
        // a refresh): with simplicity on, this generation's global champion may be a more
        // complex equal-scoring organism. Drawing the saved best keeps the network
        // consistent with the all-time-best KPI and with what a resume restores.
        // bestFitness stays per-generation: it backs the "best this generation" KPI.
        $allTimeBest = $this->best();

        return new GenerationCompleted(
            generation: $gen,
            totalGenerations: $total,
            bestFitness: $champion->fitness(),
            averageFitness: $size > 0 ? $fitnessSum / $size : 0.0,
            allTimeBestFitness: $this->bestFitness(),
            populationSize: $size,
            bestHiddenCount: $allTimeBest->hiddenCount(),
            bestGeneCount: $allTimeBest->genome()->count(),
            improved: $improved,
            durationSeconds: $duration,
            bestGenome: $allTimeBest->genome(),
            islands: array_values($stats),
            matchRate: $this->bestMatchRate,
        );
    }

    /** @return array<int, list<Organism>> islandIndex => top-K evaluated organisms */
    private function collectEmigrants(): array
    {
        $emigrants = [];
        foreach ($this->islands as $island) {
            $pop = $island->population();
            usort($pop, $this->fitter);
            $emigrants[$island->index] = array_slice($pop, 0, $this->migration->topK());
        }
        return $emigrants;
    }

    /** @param array<int, list<Organism>> $emigrants */
    private function distributeEmigrants(array $emigrants): void
    {
        $count = count($this->islands);
        foreach ($emigrants as $sourceIndex => $travellers) {
            $destIndex = $this->migration->destinationOf($sourceIndex, $count);
            if ($destIndex === $sourceIndex) {
                continue;
            }
            $dest = $this->islands[$destIndex];
            $pop = $dest->population();
            foreach ($travellers as $k => $traveller) {
                $idx = count($pop) - 1 - $k;
                if ($idx <= 0) {
                    break;
                }
                $pop[$idx] = $dest->fresh(new Organism(
                    $traveller->genome(),
                    $this->spec,
                    null,
                    new Epigenome($traveller->epigenome()->markers()),
                ));
            }
            $dest->setPopulation($pop);
        }
    }

    private function globalChampion(): Organism
    {
        $best = null;
        foreach ($this->islands as $island) {
            $champion = $island->champion();
            if ($champion !== null && ($best === null || ($this->fitter)($champion, $best) < 0)) {
                $best = $champion;
            }
        }
        return $best ?? new Organism(new Genome(), $this->spec);
    }

    private function buildIslands(Rng $master): void
    {
        $islandCount = $this->config->getIslands();
        $sizes = $this->islandSizes($islandCount, $this->config->getPopulation());

        for ($i = 0; $i < $islandCount; $i++) {
            $islandRng = $master->derive($i + 1);
            $island = new Island(
                index: $i,
                size: $sizes[$i],
                config: $this->config,
                spec: $this->spec,
                rng: $islandRng,
                ids: $this->ids,
                factory: new OrganismFactory($this->spec, $islandRng, $this->config->getInitialHidden()),
                reproducer: new Reproducer($this->config, $this->spec, $islandRng),
                selection: new TournamentSelection(),
                adaptive: $this->config->isAdaptiveMutationEnabled() ? new AdaptiveMutation(
                    $this->config->getAdaptivePatience(),
                    $this->config->getAdaptiveUpFactor(),
                    $this->config->getAdaptiveDownFactor(),
                    $this->config->getAdaptiveMinScale(),
                    $this->config->getAdaptiveMaxScale(),
                ) : null,
                trauma: $this->config->isTraumaEnabled() ? new TraumaPolicy(
                    $this->config->getTraumaIntensity(),
                    $this->config->getTraumaDecay(),
                ) : null,
            );
            $island->seed();
            $this->islands[] = $island;
        }
    }

    /** @return list<int> island sizes summing to the configured population */
    private function islandSizes(int $islandCount, int $population): array
    {
        $base = max(2, intdiv($population, $islandCount));
        $sizes = array_fill(0, $islandCount, $base);
        $remainder = $population - $base * $islandCount;
        for ($i = 0; $i < $remainder; $i++) {
            $sizes[$i]++;
        }
        return $sizes;
    }

    private function defaultEvaluator(Rng $master): FitnessEvaluator
    {
        if (!$this->config->isLifetimeLearningEnabled()) {
            return new SerialEvaluator();
        }
        $learner = new LifetimeLearner(
            $master->derive(0xA11CE),
            $this->config->getLifetimeLearningSteps(),
            $this->config->getLifetimeLearningStepSize(),
            $this->config->getLamarckianFraction(),
        );
        return new SerialEvaluator($learner);
    }

    /**
     * The whole living world as plain data, so a run can be stopped and continued
     * later: generation count, global best, and every island's genomes (with their
     * age and epigenetic markers).
     *
     * @return array<string, mixed>
     */
    public function snapshot(): array
    {
        $islands = [];
        $islandState = [];
        foreach ($this->islands as $island) {
            $organisms = [];
            foreach ($island->population() as $o) {
                $organisms[] = [
                    'genome' => $o->genome()->toArray(),
                    'markers' => $o->epigenome()->markers(),
                ];
            }
            $islands[] = $organisms;
            $islandState[] = $island->stateSnapshot();
        }

        return [
            'generation' => $this->generation,
            'bestFitness' => $this->bestFitness(),
            'bestGenome' => $this->bestGenome?->toArray() ?? [],
            'islands' => $islands,
            'islandState' => $islandState,
        ];
    }

    /** Reload a {@see snapshot()} so {@see run()} continues from where it stopped. */
    public function restore(array $snapshot): void
    {
        $this->resuming = true;
        $this->generation = (int) ($snapshot['generation'] ?? 0);
        $this->bestFitness = (float) ($snapshot['bestFitness'] ?? -INF);
        $bestGenome = $snapshot['bestGenome'] ?? [];
        $this->bestGenome = $bestGenome === [] ? null : Genome::fromArray($bestGenome);

        $saved = $snapshot['islands'] ?? [];
        $savedState = $snapshot['islandState'] ?? [];
        foreach ($this->islands as $i => $island) {
            if (isset($saved[$i]) && $saved[$i] !== []) {
                $island->seedFrom($saved[$i]);
            }
            if (isset($savedState[$i]) && is_array($savedState[$i])) {
                $island->restoreState($savedState[$i]);
            }
        }
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

    /** @return list<Organism> every organism across all islands */
    public function population(): array
    {
        $all = [];
        foreach ($this->islands as $island) {
            foreach ($island->population() as $organism) {
                $all[] = $organism;
            }
        }
        return $all;
    }

    /** @return list<Island> */
    public function islands(): array
    {
        return $this->islands;
    }

    public function config(): EvolutionConfig
    {
        return $this->config;
    }
}
