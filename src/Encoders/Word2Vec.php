<?php

namespace Rotifer\Encoders;

/**
 * This class converts words to a fixed vector based on the word hashes. Does not pay attention to meanings of them.
 */
class Word2Vec
{
	private array $vectors = [];
	private int $vectorSize;

	public function __construct($vectorSize = 32) {
		$this->vectorSize = $vectorSize;
	}

	/**
	 * Tokenize the text into individual words
	 * @param string $text
	 * @return string[]
	 */
	private function tokenize(string $text): array
	{
		// Trim
		$text = trim($text);
		// To lowercase
		$text = mb_strtolower($text, 'UTF-8');
		// Keep letters, numbers, and whitespace (exclude special characters)
		$text = preg_replace('/[^\p{L}\p{N}\s]/u', '', $text);
		// Split them by space
		return preg_split('/\s+/u', $text);
	}

	private function hashToVector($word): array
	{
		$hash = md5($word); // Hash the word using MD5
		$partLength = strlen($hash) / 2;
		$vector = [];
		for ($i = 0; $i < $this->vectorSize; $i++) {
			$charIndex = $i % $partLength;
			// Convert each part of the hash to a number and normalize
			$part = substr($hash, $charIndex * 2, 2); // Take two characters at a time
			$vector[] = (hexdec($part) / 255.0) * 2 - 1; // Normalize to [-1, 1]
		}
		return $vector;
	}

	/**
	 * Get the closest word to a given vector
	 * @param $vector
	 * @return string|null
	 */
	private function getClosestWord($vector): string|null
	{
		$closestWord = null;
		$closestDistance = PHP_FLOAT_MAX;

		foreach ($this->vectors as $word => $wordVector) {
			$distance = $this->cosineDistance($vector, $wordVector);
			if ($distance < $closestDistance) {
				$closestDistance = $distance;
				$closestWord = $word;
			}
		}

		return $closestWord;
	}

	/**
	 * Calculate cosine distance between two vectors
	 * @param $vec1
	 * @param $vec2
	 * @return float
	 */
	public function cosineDistance($vec1, $vec2): float
	{
		$dotProduct = 0.0;
		$normVec1 = 0.0;
		$normVec2 = 0.0;

		for ($i = 0; $i < count($vec1); $i++) {
			$dotProduct += $vec1[$i] * $vec2[$i];
			$normVec1 += $vec1[$i] * $vec1[$i];
			$normVec2 += $vec2[$i] * $vec2[$i];
		}

		$normVec1 = sqrt($normVec1);
		$normVec2 = sqrt($normVec2);

		if ($normVec1 == 0 || $normVec2 == 0) {
			return PHP_FLOAT_MAX;
		}

		return 1 - ($dotProduct / ($normVec1 * $normVec2));
	}

	/**
	 * Converts a word to vector
	 * @param string $word
	 * @return array
	 */
	public function word2vec(string $word): array
	{
		if (!empty($this->vectors[$word])) {
			return $this->vectors[$word];
		}

		$this->vectors[$word] = $this->hashToVector($word);

		return $this->vectors[$word];
	}

	/**
	 * Converts a sentence to array of word vectors and add eos character vector at the end
	 * @param string $text
	 * @param string $eos End of sentence character
	 * @return array[]
	 */
	public function sentence2vec(string $text, string $eos = "\n"): array
	{
		$tokens = $this->tokenize($text);

		$vectors = [];
		foreach ($tokens as $token) {
			$vectors[] = $this->word2vec($token);
		}

		// Add eos (End of sentence)
		if (!empty($eos)) {
			$vectors[] = $this->word2vec($eos);
		}

		return $vectors;
	}

	/**
	 * Converts a vector to a word
	 * @param array $vector
	 * @return string|null
	 */
	public function vec2word(array $vector): ?string
	{
		return $this->getClosestWord($vector);
	}

	/**
	 * Converts an array of vector arrays to a sentence
	 * @param array[] $vectors
	 * @param string $splitBy
	 * @return string|null
	 */
	public function vec2sentence(array $vectors, string $splitBy = ' '): ?string
	{
		$sentence = '';
		foreach ($vectors as $vector) {
			$sentence .= $this->getClosestWord($vector).$splitBy;
		}

		return trim($sentence);
	}
}
