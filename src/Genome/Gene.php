<?php

declare(strict_types=1);

namespace Rotifer\Genome;

/**
 * One connection gene: a weighted, directed edge between two neurons.
 *
 * The (from -> to) pair is the gene's identity; a genome holds at most one gene
 * per such pair, so {@see connectionKey()} doubles as a dedup/alignment key for
 * crossover and genome distance.
 */
final readonly class Gene
{
    public float $weight;

    public function __construct(
        public NodeRef $from,
        public NodeRef $to,
        float $weight,
    ) {
        $this->weight = Weight::clamp($weight);
    }

    public static function of(
        NodeType $fromType,
        int $fromIndex,
        NodeType $toType,
        int $toIndex,
        float $weight,
    ): self {
        return new self(
            new NodeRef($fromType, $fromIndex),
            new NodeRef($toType, $toIndex),
            $weight,
        );
    }

    /** Identity of the connection, ignoring weight (e.g. "0:2->1:5"). */
    public function connectionKey(): string
    {
        return $this->from->key() . '->' . $this->to->key();
    }

    public function withWeight(float $weight): self
    {
        return new self($this->from, $this->to, $weight);
    }

    /** Plain array form used by codecs and the legacy-compatible genome shape. */
    public function toArray(): array
    {
        return [
            'from_type' => $this->from->type->value,
            'from_index' => $this->from->index,
            'to_type' => $this->to->type->value,
            'to_index' => $this->to->index,
            'weight' => $this->weight,
        ];
    }

    public static function fromArray(array $a): self
    {
        return self::of(
            NodeType::from($a['from_type']),
            (int) $a['from_index'],
            NodeType::from($a['to_type']),
            (int) $a['to_index'],
            (float) $a['weight'],
        );
    }
}
