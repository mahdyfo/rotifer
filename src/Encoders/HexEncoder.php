<?php

namespace GeneticAutoml\Encoders;

class HexEncoder implements Encoder
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
        return self::binToHex(BinaryEncoder::getInstance()->encodeConnection($fromType, $fromIndex, $toType, $toIndex, $weight));
    }

    public function decodeConnection(string $encodedGene): array
    {
        return BinaryEncoder::getInstance()->decodeConnection(self::hexToBin($encodedGene));
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