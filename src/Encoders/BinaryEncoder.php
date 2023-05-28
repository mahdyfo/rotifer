<?php

namespace GeneticAutoml\Encoders;

use GeneticAutoml\Helpers\WeightHelper;
use GeneticAutoml\Models\Neuron;

class BinaryEncoder implements Encoder
{
    private static $instance;

    public static function getInstance(): self
    {
        if (!isset(self::$instance)) {
            self::$instance = new self();
        }
        return self::$instance;
    }

    /**
     * [input/neuron] [index] [neuron/output] [index] [weight]
     *      1-char    16-char       1-char    16-char 24-char
     * @param int $fromType
     * @param int $fromIndex
     * @param int $toType
     * @param int $toIndex
     * @param float $weight
     * @return string
     */
    public function encodeConnection(int $fromType, int $fromIndex, int $toType, int $toIndex, float $weight): string
    {
        // Input = 0, Neuron = 1
        $fromType = ($fromType == Neuron::TYPE_INPUT) ? 0 : 1;

        // Neuron = 0, Output = 1
        $toType = ($toType == Neuron::TYPE_OUTPUT) ? 1 : 0;

        $weight = round(($weight + WeightHelper::MAX_WEIGHT) * 1000000);

        return $fromType
            . str_pad(decbin($fromIndex), 16, 0, STR_PAD_LEFT)
            . $toType
            . str_pad(decbin($toIndex), 16, 0, STR_PAD_LEFT)
            . str_pad(decbin($weight), 24, 0, STR_PAD_LEFT);
    }

    public function decodeConnection(string $encodedGene): array
    {
        $fromType = substr($encodedGene, 0, 1) == "0" ? Neuron::TYPE_INPUT : Neuron::TYPE_HIDDEN;
        $fromIndex = substr($encodedGene, 1, 16);
        $toType = substr($encodedGene, 17, 1) == "0" ? Neuron::TYPE_HIDDEN : Neuron::TYPE_OUTPUT;
        $toIndex = substr($encodedGene, 18, 16);
        $weight = substr($encodedGene, 34, 24);

        return [
            'from_type' => $fromType,
            'from_index' => bindec($fromIndex),
            'to_type' => $toType,
            'to_index' => bindec($toIndex),
            'weight' => (bindec($weight) / 1000000) - WeightHelper::MAX_WEIGHT,
        ];
    }

    public static function binToHex(string $binaryString): string
    {
        return base_convert($binaryString, 2, 16);
    }

    public static function hexToBin(string $hexString): string
    {
        return base_convert($hexString, 16, 2);
    }
}