<?php

namespace Rotifer\Encoders;

class RandomWordEmbedding
{
    public static function hashWord(string $word, $hashAlgo = 'crc32'): array
    {
        $hashed = hash($hashAlgo, strtolower($word));
        $embedded = [];
        foreach (str_split($hashed) as $char) {
            $embedded[] = hexdec($char) / 15.0; // 15 because starts from 0 (0) to 15 (f)
        }
        return $embedded;
    }

    /**
     * @param string $sentence
     * @param string $delimiter the regex that is the splitter of words. Like \s for space chars.
     * @param string $hashAlgo
     * @param bool $withBias whether to add the value 1 at first of each word array as bias
     * @return array returns an array of embedded words
     */
    public static function hashSentence(string $sentence, string $hashAlgo = 'crc32', string $delimiter = '\s+', bool $withBias = true): array
    {
        $results = [];
        $words = preg_split('/' . $delimiter . '/', trim($sentence));
        foreach ($words as $word) {
            $embeddedWord = static::hashWord($word, $hashAlgo);
            $results[] = $withBias ? [1, ...$embeddedWord] : $embeddedWord;
        }
        return $results;
    }
}
