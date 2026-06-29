<?php

declare(strict_types=1);

namespace Rotifer\Evolution;

use Rotifer\Evolution\Adaptation\AdaptiveMutation;
use Rotifer\Evolution\Epigenetics\TraumaPolicy;
use Rotifer\Evolution\Reproduction\Reproducer;
use Rotifer\Evolution\Selection\TournamentSelection;
use Rotifer\Genome\Genome;
use Rotifer\Network\NetworkSpec;
use Rotifer\Observe\Event\IslandStat;
use Rotifer\Organism\Epigenome;
use Rotifer\Organism\Organism;
use Rotifer\Runtime\EvolutionConfig;
use Rotifer\Runtime\Rng;

/**
 * One geographic deme: a self-contained population with its own random stream,
 * adaptive-mutation and trauma state. Islands evolve semi-isolated (so they
 * drift toward different "villages") and only mix when the World migrates
 * organisms between them.
 *
 * A generation here is: rank the (already-evaluated) population, note the
 * champion, then breed the next one - applying whichever biological mechanisms
 * are switched on. Each is optional; a null collaborator means that mechanism is
 * simply skipped.
 */
final class Island
{
    /** @var list<Organism> */
    private array $population = [];

    private float $mutationScale = 1.0;
    private ?Genome $bestGenome = null;
    private float $bestFitness = -INF;
    private ?Organism $champion = null;
    private IslandStat $stat;
    /** @var callable(Organism, Organism): int */
    private $fitter;

    public function __construct(
        public readonly int $index,
        private readonly int $size,
        private readonly EvolutionConfig $config,
        private readonly NetworkSpec $spec,
        private readonly Rng $rng,
        private readonly IdSequence $ids,
        private readonly OrganismFactory $factory,
        private readonly Reproducer $reproducer,
        private readonly TournamentSelection $selection,
        private readonly ?AdaptiveMutation $adaptive = null,
        private readonly ?TraumaPolicy $trauma = null,
    ) {
        $this->stat = new IslandStat($index, 0, 0.0, 0.0);
        $this->fitter = Organism::ranker($config->getSimplicity());
    }

    public function seed(): void
    {
        $this->population = [];
        for ($i = 0; $i < $this->size; $i++) {
            $this->population[] = $this->fresh($this->factory->random());
        }
    }

    /**
     * Repopulate from saved organisms (genome + markers) so a stopped run can
     * continue. Falls back to a random seed if the snapshot is empty.
     *
     * @param list<array{genome: list<array<string,int|float>>, markers?: array<string,float>}> $saved
     */
    public function seedFrom(array $saved): void
    {
        $this->population = [];
        foreach ($saved as $entry) {
            $organism = new Organism(
                Genome::fromArray($entry['genome'] ?? []),
                $this->spec,
                null,
                new Epigenome($entry['markers'] ?? []),
            );
            $this->population[] = $this->fresh($organism);
        }
        if ($this->population === []) {
            $this->seed();
        }
    }

    /** @return list<Organism> */
    public function population(): array
    {
        return $this->population;
    }

    /** @param list<Organism> $population */
    public function setPopulation(array $population): void
    {
        $this->population = $population;
    }

    public function champion(): ?Organism
    {
        return $this->champion;
    }

    public function bestGenome(): ?Genome
    {
        return $this->bestGenome;
    }

    public function bestFitness(): float
    {
        return $this->bestFitness === -INF ? 0.0 : $this->bestFitness;
    }

    public function stat(): IslandStat
    {
        return $this->stat;
    }

    /**
     * The island's evolving state (separate from its population), so a resumed run
     * keeps its hall-of-fame best and its adaptive-mutation pressure rather than
     * resetting them - which is what made the average fitness wobble after a
     * continue.
     *
     * @return array<string, mixed>
     */
    public function stateSnapshot(): array
    {
        return [
            'mutationScale' => $this->mutationScale,
            'bestFitness' => $this->bestFitness === -INF ? null : $this->bestFitness,
            'bestGenome' => $this->bestGenome?->toArray(),
            'adaptive' => $this->adaptive?->state(),
        ];
    }

    /** @param array<string, mixed> $state */
    public function restoreState(array $state): void
    {
        $this->mutationScale = (float) ($state['mutationScale'] ?? 1.0);
        if (isset($state['bestFitness'])) {
            $this->bestFitness = (float) $state['bestFitness'];
        }
        if (!empty($state['bestGenome'])) {
            $this->bestGenome = Genome::fromArray($state['bestGenome']);
        }
        if ($this->adaptive !== null && is_array($state['adaptive'] ?? null)) {
            $this->adaptive->restore($state['adaptive']);
        }
    }

    public function fresh(Organism $organism): Organism
    {
        return $organism->withId($this->ids->next());
    }

    /** Rank the evaluated population, record the champion, and breed the next one. */
    public function reproduce(int $generation): void
    {
        $this->rankByRawFitness();
        $this->champion = $this->population[0];
        if ($this->champion->fitness() > $this->bestFitness) {
            $this->bestFitness = $this->champion->fitness();
            $this->bestGenome = $this->champion->genome();
        }

        $evaluated = $this->population;
        $this->trauma?->applyStress($evaluated);
        $this->mutationScale = $this->adaptive?->update($this->champion->fitness()) ?? 1.0;

        // Snapshot stats from the evaluated population before it is replaced.
        $this->stat = $this->buildStat($evaluated);

        $fitnessOf = $this->selectionFitness();
        $this->population = $this->breed($this->selectSurvivors(), $fitnessOf, $generation);
    }

    private function rankByRawFitness(): void
    {
        usort($this->population, $this->fitter);
    }

    /** The metric selection ranks on: raw fitness. */
    private function selectionFitness(): callable
    {
        return static fn (Organism $o): float => $o->fitness();
    }

    /** @return list<Organism> the fittest (and, on ties, simplest) survivors */
    private function selectSurvivors(): array
    {
        $candidates = $this->population;
        usort($candidates, $this->fitter);

        $count = max(2, (int) round(count($candidates) * $this->config->getSurviveRate()));
        return array_slice($candidates, 0, min($count, count($candidates)));
    }

    /**
     * @param list<Organism> $survivors
     * @param callable(Organism): float $fitnessOf
     * @return list<Organism>
     */
    private function breed(array $survivors, callable $fitnessOf, int $generation): array
    {
        $next = [];

        // Hall of fame: the island's best genome lives on as a fresh clone, so the
        // best solution is never lost from one generation to the next.
        $elitism = $this->config->getElitism();
        if ($elitism > 0 && $this->bestGenome !== null) {
            $next[] = $this->fresh(new Organism($this->bestGenome, $this->spec));
        }

        // Elitism: carry the top survivors over unchanged. They keep their
        // evaluated fitness for the rest of this generation's selection; the next
        // evaluation resets them.
        for ($i = 0; $i < $elitism - 1 && $i < count($survivors); $i++) {
            $next[] = $survivors[$i];
        }

        while (count($next) < $this->size) {
            $next[] = $this->fresh($this->breedChild($survivors, $fitnessOf));
        }

        $next = array_slice($next, 0, $this->size);
        return $this->injectDiversity($next, $generation);
    }

    /**
     * Breed one child from a parent pool: two tournament picks, crossed and
     * mutated, carrying any inherited trauma boost.
     *
     * @param list<Organism> $survivors
     * @param callable(Organism): float $fitnessOf
     */
    private function breedChild(array $survivors, callable $fitnessOf): Organism
    {
        $parentA = $this->selection->pick($survivors, $this->rng, $fitnessOf);
        $parentB = $this->selection->pick($survivors, $this->rng, $fitnessOf);

        $boost = 1.0;
        $childEpigenome = null;
        if ($this->trauma !== null) {
            $childEpigenome = $this->trauma->childEpigenome($parentA, $parentB);
            $boost = $this->trauma->mutationBoost($childEpigenome);
        }

        $child = $this->reproducer->breed($parentA, $parentB, $this->mutationScale * $boost);
        if ($childEpigenome !== null) {
            $child->setEpigenome($childEpigenome);
        }
        return $child;
    }

    /**
     * @param list<Organism> $next
     * @return list<Organism>
     */
    private function injectDiversity(array $next, int $generation): array
    {
        $rate = $this->config->getDiversityInjection();
        if ($rate <= 0.0 || $generation <= 1) {
            return $next;
        }
        $eliteHead = max(0, $this->config->getElitism());
        $fresh = max(1, (int) round(count($next) * $rate));
        for ($i = 0; $i < $fresh; $i++) {
            $idx = count($next) - 1 - $i;
            if ($idx > $eliteHead) {
                $next[$idx] = $this->fresh($this->factory->random());
            }
        }
        return $next;
    }

    /** @param list<Organism> $evaluated the population as scored this generation */
    private function buildStat(array $evaluated): IslandStat
    {
        $size = count($evaluated);
        $fitnessSum = 0.0;
        $traumaSum = 0.0;
        foreach ($evaluated as $organism) {
            $fitnessSum += $organism->fitness();
            $traumaSum += $organism->epigenome()->intensity('stress');
        }

        return new IslandStat(
            index: $this->index,
            size: $size,
            bestFitness: $this->champion?->fitness() ?? 0.0,
            averageFitness: $size > 0 ? $fitnessSum / $size : 0.0,
            mutationScale: $this->mutationScale,
            traumaLevel: $size > 0 ? $traumaSum / $size : 0.0,
        );
    }
}
