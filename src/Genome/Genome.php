<?php

declare(strict_types=1);

namespace Rotifer\Genome;

/**
 * An immutable set of connection genes - the complete genetic description of a
 * network. There is at most one gene per (from -> to) pair; insertion order is
 * preserved so serialization is stable.
 *
 * Every mutating operation returns a new Genome, leaving the original untouched
 * (so a parent genome can be safely shared across offspring and threads).
 */
final class Genome
{
    /** @var array<string, Gene> connectionKey => Gene, insertion-ordered */
    private array $genes;

    /**
     * @param iterable<Gene> $genes later genes override earlier ones on the same connection
     */
    public function __construct(iterable $genes = [])
    {
        $map = [];
        foreach ($genes as $gene) {
            $map[$gene->connectionKey()] = $gene;
        }
        $this->genes = $map;
    }

    /** @return list<Gene> */
    public function genes(): array
    {
        return array_values($this->genes);
    }

    public function count(): int
    {
        return count($this->genes);
    }

    public function isEmpty(): bool
    {
        return $this->genes === [];
    }

    public function has(string $connectionKey): bool
    {
        return isset($this->genes[$connectionKey]);
    }

    public function get(string $connectionKey): ?Gene
    {
        return $this->genes[$connectionKey] ?? null;
    }

    /** Add or replace a gene on its connection. */
    public function with(Gene $gene): self
    {
        $genes = $this->genes;
        $genes[$gene->connectionKey()] = $gene;
        return self::fromMap($genes);
    }

    /** Remove the gene on the given connection, if present. */
    public function without(string $connectionKey): self
    {
        if (!isset($this->genes[$connectionKey])) {
            return $this;
        }
        $genes = $this->genes;
        unset($genes[$connectionKey]);
        return self::fromMap($genes);
    }

    /**
     * Apply a transform to every gene, dropping any that map to null.
     * @param callable(Gene): ?Gene $fn
     */
    public function map(callable $fn): self
    {
        $out = [];
        foreach ($this->genes as $gene) {
            $mapped = $fn($gene);
            if ($mapped !== null) {
                $out[$mapped->connectionKey()] = $mapped;
            }
        }
        return self::fromMap($out);
    }

    /**
     * NEAT-style genetic distance between two genomes.
     *
     * Genes are aligned by connection identity. Genes present in only one genome
     * are "disjoint"; their share (normalized by the larger genome) plus the mean
     * absolute weight difference of the shared genes gives the distance. Larger
     * means more dissimilar - a general measure of how far two genomes have drifted.
     *
     * @param float $disjointCoefficient weight of structural difference (c1)
     * @param float $weightCoefficient   weight of average weight difference (c2)
     */
    public function distanceTo(self $other, float $disjointCoefficient = 1.0, float $weightCoefficient = 0.4): float
    {
        $disjoint = 0;
        $weightDiffSum = 0.0;
        $matching = 0;

        foreach ($this->genes as $key => $gene) {
            $otherGene = $other->genes[$key] ?? null;
            if ($otherGene === null) {
                $disjoint++;
            } else {
                $weightDiffSum += abs($gene->weight - $otherGene->weight);
                $matching++;
            }
        }
        foreach ($other->genes as $key => $gene) {
            if (!isset($this->genes[$key])) {
                $disjoint++;
            }
        }

        $largerSize = max(count($this->genes), count($other->genes));
        if ($largerSize === 0) {
            return 0.0;
        }

        $structural = $disjointCoefficient * ($disjoint / $largerSize);
        $weightual = $matching > 0
            ? $weightCoefficient * ($weightDiffSum / $matching)
            : 0.0;

        return $structural + $weightual;
    }

    /** @return list<array<string,int|float>> legacy-compatible gene arrays */
    public function toArray(): array
    {
        return array_map(static fn (Gene $g) => $g->toArray(), $this->genes());
    }

    /** @param iterable<array<string,int|float>> $rows */
    public static function fromArray(iterable $rows): self
    {
        $genes = [];
        foreach ($rows as $row) {
            $genes[] = Gene::fromArray($row);
        }
        return new self($genes);
    }

    /** @param array<string, Gene> $map */
    private static function fromMap(array $map): self
    {
        $genome = new self();
        $genome->genes = $map;
        return $genome;
    }
}
