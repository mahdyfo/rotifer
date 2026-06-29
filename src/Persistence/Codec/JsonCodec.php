<?php

declare(strict_types=1);

namespace Rotifer\Persistence\Codec;

use Rotifer\Genome\Gene;
use Rotifer\Genome\Genome;
use Rotifer\Genome\NodeType;

/**
 * Readable JSON encoding: each gene is a compact tuple
 * [fromType, fromIndex, toType, toIndex, weight]. Used by the web dashboard,
 * replay files and debugging, where legibility beats compactness.
 */
final class JsonCodec implements Codec
{
    public function encode(Genome $genome): string
    {
        $tuples = array_map(
            fn (Gene $g) => $this->tuple($g),
            $genome->genes(),
        );
        return json_encode($tuples, JSON_THROW_ON_ERROR);
    }

    public function decode(string $encoded): Genome
    {
        $tuples = json_decode($encoded, true, 512, JSON_THROW_ON_ERROR);
        $genes = array_map(fn (array $t) => $this->fromTuple($t), $tuples);
        return new Genome($genes);
    }

    public function encodeGene(Gene $gene): string
    {
        return json_encode($this->tuple($gene), JSON_THROW_ON_ERROR);
    }

    public function decodeGene(string $encoded): Gene
    {
        return $this->fromTuple(json_decode($encoded, true, 512, JSON_THROW_ON_ERROR));
    }

    private function tuple(Gene $gene): array
    {
        return [
            $gene->from->type->value,
            $gene->from->index,
            $gene->to->type->value,
            $gene->to->index,
            $gene->weight,
        ];
    }

    private function fromTuple(array $t): Gene
    {
        return Gene::of(
            NodeType::from((int) $t[0]),
            (int) $t[1],
            NodeType::from((int) $t[2]),
            (int) $t[3],
            (float) $t[4],
        );
    }
}
