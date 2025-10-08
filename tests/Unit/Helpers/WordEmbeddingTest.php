<?php

namespace Rotifer\Tests\Unit\Helpers;

use PHPUnit\Framework\TestCase;
use Rotifer\Encoders\WordEmbedding;

class WordEmbeddingTest extends TestCase
{
    public function testConstructorWithValidDimensions()
    {
        $embedding = new WordEmbedding(10);
        $this->assertEquals(10, $embedding->getDimensions());

        $embedding = new WordEmbedding(1);
        $this->assertEquals(1, $embedding->getDimensions());

        $embedding = new WordEmbedding(100);
        $this->assertEquals(100, $embedding->getDimensions());
    }

    public function testConstructorWithInvalidDimensions()
    {
        $this->expectException(\InvalidArgumentException::class);
        new WordEmbedding(0);
    }

    public function testConstructorWithNegativeDimensions()
    {
        $this->expectException(\InvalidArgumentException::class);
        new WordEmbedding(-5);
    }

    public function testEmbedReturnsCorrectDimensions()
    {
        $embedding = new WordEmbedding(10);
        $vector = $embedding->embed("hello");
        $this->assertCount(10, $vector);

        $embedding = new WordEmbedding(50);
        $vector = $embedding->embed("world");
        $this->assertCount(50, $vector);
    }

    public function testEmbedReturnsFloats()
    {
        $embedding = new WordEmbedding(10);
        $vector = $embedding->embed("test");

        foreach ($vector as $value) {
            $this->assertIsFloat($value);
        }
    }

    public function testEmbedValuesInRange()
    {
        $embedding = new WordEmbedding(20);
        $vector = $embedding->embed("test");

        foreach ($vector as $value) {
            $this->assertGreaterThanOrEqual(-1, $value);
            $this->assertLessThanOrEqual(1, $value);
        }
    }

    public function testEmbedIsDeterministic()
    {
        $embedding = new WordEmbedding(10);
        $vector1 = $embedding->embed("hello");
        $vector2 = $embedding->embed("hello");

        $this->assertEquals($vector1, $vector2);
    }

    public function testEmbedIsDeterministicAcrossInstances()
    {
        $embedding1 = new WordEmbedding(10);
        $embedding2 = new WordEmbedding(10);

        $vector1 = $embedding1->embed("hello");
        $vector2 = $embedding2->embed("hello");

        $this->assertEquals($vector1, $vector2);
    }

    public function testDifferentWordsProduceDifferentVectors()
    {
        $embedding = new WordEmbedding(10);
        $vector1 = $embedding->embed("hello");
        $vector2 = $embedding->embed("world");

        $this->assertNotEquals($vector1, $vector2);
    }

    public function testCaseInsensitive()
    {
        $embedding = new WordEmbedding(10);
        $vector1 = $embedding->embed("Hello");
        $vector2 = $embedding->embed("hello");
        $vector3 = $embedding->embed("HELLO");

        $this->assertEquals($vector1, $vector2);
        $this->assertEquals($vector2, $vector3);
    }

    public function testTrimsWhitespace()
    {
        $embedding = new WordEmbedding(10);
        $vector1 = $embedding->embed("hello");
        $vector2 = $embedding->embed("  hello  ");
        $vector3 = $embedding->embed("\thello\n");

        $this->assertEquals($vector1, $vector2);
        $this->assertEquals($vector2, $vector3);
    }

    public function testEmbedManyReturnsCorrectCount()
    {
        $embedding = new WordEmbedding(10);
        $words = ["hello", "world", "test"];
        $vectors = $embedding->embedMany($words);

        $this->assertCount(3, $vectors);
        $this->assertCount(10, $vectors[0]);
        $this->assertCount(10, $vectors[1]);
        $this->assertCount(10, $vectors[2]);
    }

    public function testEmbedManyProducesSameResultsAsIndividualEmbeds()
    {
        $embedding = new WordEmbedding(10);
        $words = ["hello", "world", "test"];

        $vectorsMany = $embedding->embedMany($words);
        $vectorsIndividual = [
            $embedding->embed("hello"),
            $embedding->embed("world"),
            $embedding->embed("test"),
        ];

        $this->assertEquals($vectorsIndividual, $vectorsMany);
    }

    public function testEmbedManyWithEmptyArray()
    {
        $embedding = new WordEmbedding(10);
        $vectors = $embedding->embedMany([]);

        $this->assertCount(0, $vectors);
    }

    public function testCacheWorks()
    {
        $embedding = new WordEmbedding(10);

        // First call should cache
        $vector1 = $embedding->embed("hello");

        // Second call should use cache (same reference)
        $vector2 = $embedding->embed("hello");

        $this->assertSame($vector1, $vector2);
    }

    public function testClearCache()
    {
        $embedding = new WordEmbedding(10);

        $vector1 = $embedding->embed("hello");
        $embedding->clearCache();
        $vector2 = $embedding->embed("hello");

        // Values should be equal
        $this->assertEquals($vector1, $vector2);
    }

    public function testDifferentDimensionsProduceDifferentVectors()
    {
        $embedding10 = new WordEmbedding(10);
        $embedding20 = new WordEmbedding(20);

        $vector10 = $embedding10->embed("hello");
        $vector20 = $embedding20->embed("hello");

        $this->assertCount(10, $vector10);
        $this->assertCount(20, $vector20);
    }

    public function testEmptyStringEmbedding()
    {
        $embedding = new WordEmbedding(10);
        $vector = $embedding->embed("");

        $this->assertCount(10, $vector);
        foreach ($vector as $value) {
            $this->assertIsFloat($value);
            $this->assertGreaterThanOrEqual(-1, $value);
            $this->assertLessThanOrEqual(1, $value);
        }
    }

    public function testSpecialCharacters()
    {
        $embedding = new WordEmbedding(10);
        $vector1 = $embedding->embed("hello!");
        $vector2 = $embedding->embed("hello?");
        $vector3 = $embedding->embed("hello");

        $this->assertCount(10, $vector1);
        $this->assertCount(10, $vector2);
        $this->assertNotEquals($vector1, $vector2);
        $this->assertNotEquals($vector1, $vector3);
    }

    public function testUnicodeCharacters()
    {
        $embedding = new WordEmbedding(10);
        $vector1 = $embedding->embed("café");
        $vector2 = $embedding->embed("naïve");
        $vector3 = $embedding->embed("日本語");

        $this->assertCount(10, $vector1);
        $this->assertCount(10, $vector2);
        $this->assertCount(10, $vector3);

        foreach ([$vector1, $vector2, $vector3] as $vector) {
            foreach ($vector as $value) {
                $this->assertIsFloat($value);
                $this->assertGreaterThanOrEqual(-1, $value);
                $this->assertLessThanOrEqual(1, $value);
            }
        }
    }

    public function testDecodeReturnsNullForEmptyVocabulary()
    {
        $embedding = new WordEmbedding(10);
        $vector = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

        $this->assertNull($embedding->decode($vector));
    }

    public function testDecodeReturnsExactMatch()
    {
        $embedding = new WordEmbedding(10);
        $embedding->embed("hello");
        $embedding->embed("world");

        $helloVector = $embedding->embed("hello");
        $decoded = $embedding->decode($helloVector);

        $this->assertEquals("hello", $decoded);
    }

    public function testDecodeReturnsClosestMatch()
    {
        $embedding = new WordEmbedding(10);
        $embedding->embed("hello");
        $embedding->embed("world");
        $embedding->embed("test");

        $helloVector = $embedding->embed("hello");

        // Slightly modify the vector
        $modifiedVector = $helloVector;
        $modifiedVector[0] += 0.01;
        $modifiedVector[1] -= 0.01;

        $decoded = $embedding->decode($modifiedVector);

        $this->assertEquals("hello", $decoded);
    }

    public function testGetVocabulary()
    {
        $embedding = new WordEmbedding(10);
        $embedding->embed("hello");
        $embedding->embed("world");
        $embedding->embed("test");

        $vocab = $embedding->getVocabulary();

        $this->assertCount(3, $vocab);
        $this->assertContains("hello", $vocab);
        $this->assertContains("world", $vocab);
        $this->assertContains("test", $vocab);
    }

    public function testGetVocabularyNoDuplicates()
    {
        $embedding = new WordEmbedding(10);
        $embedding->embed("hello");
        $embedding->embed("hello");
        $embedding->embed("world");

        $vocab = $embedding->getVocabulary();

        $this->assertCount(2, $vocab);
    }

    public function testClearCacheClearsVocabulary()
    {
        $embedding = new WordEmbedding(10);
        $embedding->embed("hello");
        $embedding->embed("world");

        $this->assertCount(2, $embedding->getVocabulary());

        $embedding->clearCache();

        $this->assertCount(0, $embedding->getVocabulary());
    }

    public function testDecodeWithDifferentDimensions()
    {
        $embedding = new WordEmbedding(10);
        $embedding->embed("hello");

        // Vector with wrong dimensions (5 instead of 10)
        $shortVector = [0.1, 0.2, 0.3, 0.4, 0.5];
        $decoded = $embedding->decode($shortVector);

        // Should still return closest match using available dimensions
        $this->assertNotNull($decoded);
    }
}