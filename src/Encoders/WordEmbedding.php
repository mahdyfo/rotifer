<?php

namespace Rotifer\Encoders;

class WordEmbedding
{
    public static function hash(string $word, $hashAlgo = 'md5'): array
    {
        $hashed = hash($hashAlgo, $word);
        $embedded = [];
        foreach (str_split($hashed) as $char) {
            $embedded[] = hexdec($char) / 15.0; // 15 because starts from 0 (0) to 15 (f)
        }
        return $embedded;
    }
}
