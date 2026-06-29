<?php

declare(strict_types=1);

namespace Rotifer\Persistence\Codec;

use Rotifer\Genome\Gene;
use Rotifer\Genome\Genome;
use Rotifer\Genome\NodeType;
use Rotifer\Genome\Weight;

/**
 * Compact fixed-width binary encoding: 10 bytes per gene, so a genome is just
 * the records concatenated (no separator needed). Layout per gene:
 *
 *   1 byte  fromType        (C)
 *   2 bytes fromIndex       (n, uint16 big-endian)
 *   1 byte  toType          (C)
 *   2 bytes toIndex         (n)
 *   4 bytes weight          (N, int32 two's-complement, scaled by Weight::SCALE)
 */
final class BinaryCodec implements Codec
{
    public const RECORD_BYTES = 10;

    public function encode(Genome $genome): string
    {
        $out = '';
        foreach ($genome->genes() as $gene) {
            $out .= $this->encodeGene($gene);
        }
        return $out;
    }

    public function decode(string $encoded): Genome
    {
        $genes = [];
        $length = strlen($encoded);
        for ($offset = 0; $offset + self::RECORD_BYTES <= $length; $offset += self::RECORD_BYTES) {
            $genes[] = $this->decodeGene(substr($encoded, $offset, self::RECORD_BYTES));
        }
        return new Genome($genes);
    }

    public function encodeGene(Gene $gene): string
    {
        $scaled = (int) round(Weight::clamp($gene->weight) * Weight::SCALE);
        return pack(
            'CnCnN',
            $gene->from->type->value,
            $gene->from->index,
            $gene->to->type->value,
            $gene->to->index,
            $scaled & 0xFFFFFFFF, // two's-complement into an unsigned slot
        );
    }

    public function decodeGene(string $encoded): Gene
    {
        $f = unpack('CfromType/nfromIndex/CtoType/ntoIndex/Nweight', $encoded);
        $scaled = $f['weight'];
        if ($scaled >= 0x80000000) {
            $scaled -= 0x100000000; // restore the sign
        }
        return Gene::of(
            NodeType::from($f['fromType']),
            $f['fromIndex'],
            NodeType::from($f['toType']),
            $f['toIndex'],
            $scaled / Weight::SCALE,
        );
    }
}
