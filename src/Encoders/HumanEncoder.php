<?php

namespace GeneticAutoml\Encoders;

use GeneticAutoml\Models\Neuron;

class HumanEncoder implements Encoder
{
    private static $instance;

    public static function getInstance(): self
    {
        if (!isset(self::$instance)) {
            self::$instance = new self();
        }
        return self::$instance;
    }

    public function encodeConnection(int $fromType, int $fromIndex, int $toType, int $toIndex, float $weight): string
    {
        return ' From ' . ($fromType == Neuron::TYPE_INPUT ? 'input' : 'neuron') . ' ' . $fromIndex . ' to ' . ($toType == Neuron::TYPE_HIDDEN ? 'neuron' : 'output') . ' ' . $toIndex . ' weight ' . $weight . ' ';
    }

    public function decodeConnection(string $encodedGene): array
    {
        // No need for decode. It is a human-readable encoder
        return [];
    }
}