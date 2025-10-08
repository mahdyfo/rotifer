<?php

namespace Rotifer\Encoders;

/**
 * WordEmbedding class converts words to numerical vectors
 *
 * Uses a hash-based approach to generate consistent embeddings for words.
 * Each word is mapped to a fixed-length vector of floats in range [-1, 1].
 */
class WordEmbedding
{
    private int $dimensions;
    private array $cache = [];
    private array $vocabulary = [];

    /**
     * @param int $dimensions Number of dimensions for the embedding vector
     */
    public function __construct(int $dimensions = 10)
    {
        if ($dimensions < 1) {
            throw new \InvalidArgumentException("Dimensions must be at least 1");
        }
        $this->dimensions = $dimensions;
    }

    /**
     * Convert a word to a numerical vector
     *
     * @param string $word The word to embed
     * @return array Array of floats in range [-1, 1]
     */
    public function embed(string $word): array
    {
        $word = strtolower(trim($word));

        if (isset($this->cache[$word])) {
            return $this->cache[$word];
        }

        $embedding = [];
        for ($i = 0; $i < $this->dimensions; $i++) {
            $hash = hash('sha256', $word . '_' . $i);
            $intValue = hexdec(substr($hash, 0, 8));
            $normalized = ($intValue / 0xFFFFFFFF) * 2 - 1;
            $embedding[] = $normalized;
        }

        $this->cache[$word] = $embedding;
        $this->vocabulary[] = $word;
        return $embedding;
    }

    /**
     * Convert multiple words to their embeddings
     *
     * @param array $words Array of words to embed
     * @return array Array of embedding vectors
     */
    public function embedMany(array $words): array
    {
        return array_map(fn($word) => $this->embed($word), $words);
    }

    /**
     * Get the number of dimensions
     *
     * @return int
     */
    public function getDimensions(): int
    {
        return $this->dimensions;
    }

    /**
     * Find the closest word to a given vector
     *
     * @param array $vector The vector to decode
     * @return string|null The closest word or null if vocabulary is empty
     */
    public function decode(array $vector): ?string
    {
        if (empty($this->vocabulary)) {
            return null;
        }

        $closestWord = null;
        $minDistance = PHP_FLOAT_MAX;

        foreach ($this->vocabulary as $word) {
            $embedding = $this->cache[$word];
            $distance = $this->euclideanDistance($vector, $embedding);

            if ($distance < $minDistance) {
                $minDistance = $distance;
                $closestWord = $word;
            }
        }

        return $closestWord;
    }

    /**
     * Calculate Euclidean distance between two vectors
     *
     * @param array $vector1
     * @param array $vector2
     * @return float
     */
    private function euclideanDistance(array $vector1, array $vector2): float
    {
        $sum = 0;
        $count = min(count($vector1), count($vector2));

        for ($i = 0; $i < $count; $i++) {
            $diff = $vector1[$i] - $vector2[$i];
            $sum += $diff * $diff;
        }

        return sqrt($sum);
    }

    /**
     * Get all words in the vocabulary
     *
     * @return array
     */
    public function getVocabulary(): array
    {
        return $this->vocabulary;
    }

    /**
     * Clear the embedding cache
     *
     * @return void
     */
    public function clearCache(): void
    {
        $this->cache = [];
        $this->vocabulary = [];
    }
}