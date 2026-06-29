<?php

declare(strict_types=1);

namespace Rotifer\Persistence\Codec;

use Rotifer\Genome\Gene;
use Rotifer\Genome\Genome;

/**
 * Serializes a genome to and from a string. Implementations choose their own
 * framing (fixed-width records or a separator) so {@see encode()}/{@see decode()}
 * always round-trip without a caller-supplied separator.
 */
interface Codec
{
    public function encode(Genome $genome): string;

    public function decode(string $encoded): Genome;

    public function encodeGene(Gene $gene): string;

    public function decodeGene(string $encoded): Gene;
}
