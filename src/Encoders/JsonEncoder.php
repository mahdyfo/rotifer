<?php

namespace Rotifer\Encoders;

class JsonEncoder implements Encoder
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
        return json_encode([$fromType, $fromIndex, $toType, $toIndex, $weight]);
    }

    public function decodeConnection(string $encodedGene): array
    {
        $decoded = json_decode($encodedGene);

        return [
            'from_type' => $decoded[0],
            'from_index' => $decoded[1],
            'to_type' => $decoded[2],
            'to_index' => $decoded[3],
            'weight' => $decoded[4],
        ];
    }
}