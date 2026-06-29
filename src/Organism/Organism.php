<?php

declare(strict_types=1);

namespace Rotifer\Organism;

use Rotifer\Genome\Genome;
use Rotifer\Network\Brain;
use Rotifer\Network\NetworkSpec;

/**
 * A living individual: a genome compiled into a runnable {@see Brain}, plus the
 * scratch state evolution cares about (fitness, identity).
 *
 * The genome is pruned when the brain is built, so {@see genome()} always
 * returns the effective wiring - which is exactly what reproduction should
 * inherit. Genome and network spec are fixed for an organism's life; fitness is
 * the only thing that changes as it is evaluated and survives.
 */
final class Organism
{
    private Brain $brain;
    private float $fitness = 0.0;
    private Epigenome $epigenome;

    public function __construct(
        Genome $genome,
        private readonly NetworkSpec $spec,
        private ?string $id = null,
        ?Epigenome $epigenome = null,
    ) {
        $this->brain = new Brain($genome, $spec);
        $this->epigenome = $epigenome ?? new Epigenome();
    }

    /**
     * Replace this organism's genome and recompile its brain in place - how
     * within-lifetime learning writes its inherited (Lamarckian) gains back onto
     * the individual without disturbing the population structure.
     */
    public function adoptGenome(Genome $genome): self
    {
        $this->brain = new Brain($genome, $this->spec);
        return $this;
    }

    public function epigenome(): Epigenome
    {
        return $this->epigenome;
    }

    public function setEpigenome(Epigenome $epigenome): self
    {
        $this->epigenome = $epigenome;
        return $this;
    }

    public function genome(): Genome
    {
        return $this->brain->genome();
    }

    public function spec(): NetworkSpec
    {
        return $this->spec;
    }

    public function brain(): Brain
    {
        return $this->brain;
    }

    /** Feed one input vector through the network. Returns $this for chaining. */
    public function step(array $inputs): self
    {
        $this->brain->step($inputs);
        return $this;
    }

    /** @return list<float> */
    public function outputs(): array
    {
        return $this->brain->outputs();
    }

    /** Clear network memory and zero fitness, before a fresh evaluation. */
    public function reset(): self
    {
        $this->brain->resetMemory();
        $this->fitness = 0.0;
        return $this;
    }

    /** Clear only network memory (between sequences within one evaluation). */
    public function resetMemory(): self
    {
        $this->brain->resetMemory();
        return $this;
    }

    public function fitness(): float
    {
        return $this->fitness;
    }

    public function setFitness(float $fitness): self
    {
        $this->fitness = $fitness;
        return $this;
    }

    public function addFitness(float $delta): self
    {
        $this->fitness += $delta;
        return $this;
    }

    public function id(): ?string
    {
        return $this->id;
    }

    public function withId(string $id): self
    {
        $this->id = $id;
        return $this;
    }

    public function hiddenCount(): int
    {
        return $this->brain->hiddenCount();
    }

    /**
     * Rank comparator (usort) for a given simplicity tolerance, in significant figures
     * $significantFigures > 0: fitness is rounded to that many sig-figs, so negligibly
     * different scores tie and the simpler network wins (fewer hidden, then fewer
     * connections), then lower id - this stops a solved problem from bloating
     * $significantFigures = 0: pure exact-fitness ranking, no simplicity bias, so
     * topology evolves as far as it can regardless of complexity
     */
    public static function ranker(int $significantFigures = 3): callable
    {
        return static function (self $a, self $b) use ($significantFigures): int {
            $fa = $significantFigures > 0 ? self::roundSig($a->fitness, $significantFigures) : $a->fitness;
            $fb = $significantFigures > 0 ? self::roundSig($b->fitness, $significantFigures) : $b->fitness;
            if ($fa !== $fb) {
                return $fb <=> $fa;
            }
            if ($significantFigures > 0) {
                if ($a->hiddenCount() !== $b->hiddenCount()) {
                    return $a->hiddenCount() <=> $b->hiddenCount();
                }
                $genesA = $a->genome()->count();
                $genesB = $b->genome()->count();
                if ($genesA !== $genesB) {
                    return $genesA <=> $genesB;
                }
            }
            return (int) $a->id() <=> (int) $b->id();
        };
    }

    // round to $sigfigs significant figures, a deterministic key so ranking stays a total order
    private static function roundSig(float $fitness, int $sigfigs): float
    {
        if ($fitness === 0.0) {
            return 0.0;
        }
        $factor = 10 ** (floor(log10(abs($fitness))) - ($sigfigs - 1));
        return round($fitness / $factor) * $factor;
    }
}
