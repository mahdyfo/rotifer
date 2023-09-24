<?php

namespace Rotifer\GeneEncoders;

interface Encoder
{
    public function encodeConnection(int $fromType, int $fromIndex, int $toType, int $toIndex, float $weight): string;
    public function decodeConnection(string $encodedGene): array;
}