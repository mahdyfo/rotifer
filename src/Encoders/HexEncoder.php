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
        return self::baseConvert($binaryString, 2, 16);
    }

    public static function hexToBin(string $hexString): string
    {
        return self::baseConvert($hexString, 16, 2);
    }

    /**
     * Base convert large numbers
     * https://www.php.net/manual/en/function.base-convert.php#127539
     * @param $numString
     * @param $fromBase
     * @param $toBase
     * @return string
     */
    public static function baseConvert($numString, $fromBase, $toBase): string
    {
        $chars = "0123456789abcdefghijklmnopqrstuvwxyz";
        $tostring = substr($chars, 0, $toBase);

        $length = strlen($numString);
        $result = '';
        for ($i = 0; $i < $length; $i++)
        {
            $number[$i] = strpos($chars, $numString[$i]);
        }
        do
        {
            $divide = 0;
            $newlen = 0;
            for ($i = 0; $i < $length; $i++)
            {
                $divide = $divide * $fromBase + $number[$i];
                if ($divide >= $toBase)
                {
                    $number[$newlen++] = (int)($divide / $toBase);
                    $divide = $divide % $toBase;
                } elseif ($newlen > 0)
                {
                    $number[$newlen++] = 0;
                }
            }
            $length = $newlen;
            $result = $tostring[$divide] . $result;
        } while ($newlen != 0);
        return $result;
    }
}