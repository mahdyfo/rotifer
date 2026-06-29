<?php

declare(strict_types=1);

namespace Rotifer\Runtime\Parallel;

use Amp\Cancellation;
use Amp\Parallel\Worker\Task;
use Amp\Sync\Channel;
use Rotifer\Evolution\Adaptation\AdaptiveMutation;
use Rotifer\Evolution\Epigenetics\TraumaPolicy;
use Rotifer\Evolution\IdSequence;
use Rotifer\Evolution\Island;
use Rotifer\Evolution\OrganismFactory;
use Rotifer\Evolution\Reproduction\Reproducer;
use Rotifer\Evolution\Selection\TournamentSelection;
use Rotifer\Network\NetworkSpec;
use Rotifer\Organism\Organism;
use Rotifer\Runtime\EvolutionConfig;
use Rotifer\Runtime\Fitness\Scorer;
use Rotifer\Runtime\Rng;

/**
 * Evolve one island for a whole migration epoch (K generations) inside a worker.
 *
 * This is the coarse-grained alternative to {@see EvaluationTask}: instead of
 * shipping every organism every generation just to score it, the island's
 * population crosses the boundary once per epoch, runs all K evaluate+reproduce
 * cycles locally, and only its champion, migrants and resulting population come
 * back. Communication drops from O(pop x gens) to O(pop) per epoch.
 *
 * Determinism is per-epoch: the island's Rng is re-derived from the master seed,
 * island index and epoch number, so the same seed replays the same parallel run
 * (it is not bit-identical to a serial run - that is the documented trade-off).
 *
 * @phpstan-type PopSnapshot list<array{genome: list<array<string,int|float>>, markers: array<string,float>}>
 * @implements Task<array{index:int, population:array, state:array, migrants:list<list<array<string,int|float>>>, stat:array{size:int,best:float,avg:float,mutationScale:float,trauma:float}, bestFitness:float, bestGenome:list<array<string,int|float>>, hidden:int, genes:int}, never, never>
 */
final class IslandEpochTask implements Task
{
    /**
     * @param class-string $problemClass
     * @param array{populations?: PopSnapshot, state?: array<string,mixed>}|array<string,mixed> $islandData
     */
    public function __construct(
        private readonly string $problemClass,
        private readonly NetworkSpec $spec,
        private readonly EvolutionConfig $config,
        private readonly int $islandIndex,
        private readonly int $size,
        private readonly int $masterSeed,
        private readonly int $epoch,
        private readonly int $startGeneration,
        private readonly int $generations,
        private readonly array $population, // PopSnapshot; [] on the first epoch -> random seed
        private readonly array $state,      // Island::stateSnapshot() from the previous epoch; [] on the first
        private readonly int $migrants,     // how many top genomes to return for migration
    ) {
    }

    public function run(Channel $channel, Cancellation $cancellation): array
    {
        $problem = new $this->problemClass();
        $config = $this->config;
        // Per-epoch deterministic stream, derived once from the master with a combined
        // island+epoch key (chaining derive() would overflow PHP's int range into floats).
        $stream = ($this->islandIndex + 1) * 1_000_003 + ($this->epoch + 1);
        $rng = (new Rng($this->masterSeed))->derive($stream);

        $island = new Island(
            index: $this->islandIndex,
            size: $this->size,
            config: $config,
            spec: $this->spec,
            rng: $rng,
            ids: new IdSequence(),
            factory: new OrganismFactory($this->spec, $rng, $config->getInitialHidden()),
            reproducer: new Reproducer($config, $this->spec, $rng),
            selection: new TournamentSelection(),
            adaptive: $config->isAdaptiveMutationEnabled() ? new AdaptiveMutation(
                $config->getAdaptivePatience(),
                $config->getAdaptiveUpFactor(),
                $config->getAdaptiveDownFactor(),
                $config->getAdaptiveMinScale(),
                $config->getAdaptiveMaxScale(),
            ) : null,
            trauma: $config->isTraumaEnabled() ? new TraumaPolicy(
                $config->getTraumaIntensity(),
                $config->getTraumaDecay(),
            ) : null,
        );
        $island->seedFrom($this->population); // empty -> random seed (deterministic via $rng)
        if ($this->state !== []) {
            $island->restoreState($this->state);
        }

        $migrants = [];
        $last = $this->startGeneration + $this->generations;
        for ($g = $this->startGeneration + 1; $g <= $last; $g++) {
            foreach ($island->population() as $organism) {
                $organism->setFitness(Scorer::score($organism, $problem));
            }
            if ($g === $last) {
                $migrants = $this->topGenomes($island->population(), $this->migrants);
            }
            $island->reproduce($g);
        }

        $stat = $island->stat();
        $best = $island->bestGenome();
        return [
            'index' => $this->islandIndex,
            'population' => $this->snapshotPopulation($island->population()),
            'state' => $island->stateSnapshot(),
            'migrants' => $migrants,
            'stat' => [
                'size' => $stat->size,
                'best' => $stat->bestFitness,
                'avg' => $stat->averageFitness,
                'mutationScale' => $stat->mutationScale,
                'trauma' => $stat->traumaLevel,
            ],
            'bestFitness' => $island->bestFitness(),
            'bestGenome' => $best?->toArray() ?? [],
            'hidden' => $best === null ? 0 : (new \Rotifer\Network\Brain($best, $this->spec))->hiddenCount(),
            'genes' => $best?->count() ?? 0,
        ];
    }

    /**
     * @param list<Organism> $population already scored
     * @return list<list<array<string,int|float>>> the top $k genomes by fitness
     */
    private function topGenomes(array $population, int $k): array
    {
        if ($k <= 0) {
            return [];
        }
        $ranked = $population;
        usort($ranked, Organism::ranker($this->config->getSimplicity()));
        $out = [];
        foreach (array_slice($ranked, 0, $k) as $organism) {
            $out[] = $organism->genome()->toArray();
        }
        return $out;
    }

    /**
     * @param list<Organism> $population
     * @return list<array{genome: list<array<string,int|float>>, markers: array<string,float>}>
     */
    private function snapshotPopulation(array $population): array
    {
        $out = [];
        foreach ($population as $organism) {
            $out[] = [
                'genome' => $organism->genome()->toArray(),
                'markers' => $organism->epigenome()->markers(),
            ];
        }
        return $out;
    }
}
