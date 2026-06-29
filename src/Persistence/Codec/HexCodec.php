<?php

declare(strict_types=1);

namespace Rotifer\Persistence\Codec;

use Rotifer\Genome\Gene;
use Rotifer\Genome\Genome;

/**
 * Hexadecimal view of {@see BinaryCodec}: 20 hex chars per gene, text-safe and
 * still compact. The default format for on-disk autosaves.
 */
final class HexCodec implements Codec
{
    private const HEX_PER_GENE = BinaryCodec::RECORD_BYTES * 2;

    public function __construct(private readonly BinaryCodec $binary = new BinaryCodec())
    {
    }

    public function encode(Genome $genome): string
    {
        return bin2hex($this->binary->encode($genome));
    }

    public function decode(string $encoded): Genome
    {
        return $this->binary->decode(hex2bin($encoded));
    }

    public function encodeGene(Gene $gene): string
    {
        return bin2hex($this->binary->encodeGene($gene));
    }

    public function decodeGene(string $encoded): Gene
    {
        return $this->binary->decodeGene(hex2bin($encoded));
    }
}
