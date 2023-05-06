<?php

namespace GeneticAutoml\Encoders;

interface Encoder
{
    public function encodeConnection(int $fromType, $fromIndex, $toType, $toIndex, $weight): string;
    public function decodeConnection(string $binaryGene): array;
}