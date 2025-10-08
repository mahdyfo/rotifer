<?php

require 'vendor/autoload.php';

use Rotifer\Models\{Agent, World};
use Rotifer\Activations\Activation;
use Rotifer\Encoders\WordEmbedding;

// Evolution constants
const PROBABILITY_CROSSOVER = 0.5;
const PROBABILITY_MUTATE_WEIGHT = 0.4;
const MUTATE_WEIGHT_COUNT = 2;
const PROBABILITY_MUTATE_WEIGHT_RANDOM = 0.95;
const PROBABILITY_MUTATE_ADD_NEURON = 0.1;
const PROBABILITY_MUTATE_REMOVE_NEURON = 0.1;
const PROBABILITY_MUTATE_ADD_GENE = 0.1;
const PROBABILITY_MUTATE_REMOVE_GENE = 0.1;
const ACTIVATION = [Activation::class, 'tanh'];
const SAVE_WORLD_EVERY_GENERATION = 0;
const CALCULATE_STEP_TIME = false;
const ONLY_CALCULATE_FIRST_STEP_TIME = false;
const DIVERSITY_INJECTION_RATE = 0.05; // 5% of population replaced with fresh agents each generation
const ELITISM_COUNT = 1; // Keep top N agents guaranteed to survive (0 to disable)

// Configuration
$embeddingDimensions = 3;
$populationSize = 120;
$generations = 1000;
$survivalRate = 0.2;

// Training data: [question, answer]
// Start with simpler, shorter responses to make learning easier
$trainingData = [
    ["hello man", "hello matt"],
    ["hi joe", "hi"],
    ["bye john", "bye"],
    ["thanks", "welcome"],
    ["name", "rotifer"],
    ["who are you", "rotifer"],
];

echo "=== Rotifer Chatbot Example ===\n\n";
echo "Training a simple question-answering chatbot using word embeddings...\n";
echo "Embedding dimensions: $embeddingDimensions\n";
echo "Population size: $populationSize\n";
echo "Generations per training: $generations\n\n";

// Initialize word embedding
$embedding = new WordEmbedding($embeddingDimensions);

// Build vocabulary from all questions and answers
echo "Building vocabulary...\n";
$allWords = [];
foreach ($trainingData as [$question, $answer]) {
    $questionWords = explode(' ', $question);
    $answerWords = explode(' ', $answer);
    $allWords = array_merge($allWords, $questionWords, $answerWords);
}
$allWords = array_unique($allWords);

foreach ($allWords as $word) {
    $embedding->embed($word);
}
echo "Vocabulary size: " . count($embedding->getVocabulary()) . " words\n\n";

// Special tokens
$embedding->embed('<not_started>');
$embedding->embed('<end_of_sentence>');

// Convert training data to sequential format
echo "Converting training data to sequential format...\n";
echo "Example for first training pair:\n";
echo "Q: \"{$trainingData[0][0]}\"\n";
echo "A: \"{$trainingData[0][1]}\"\n\n";
echo "Sequential data format:\n";

$data = [];
$showExample = true;
foreach ($trainingData as $idx => [$question, $answer]) {
    $questionWords = explode(' ', $question);
    $answerWords = explode(' ', $answer);

    if ($showExample) {
        echo "  [Input] -> [Expected Output]\n";
    }

    // Phase 1: Process question words, output <not_started>
    foreach ($questionWords as $word) {
        if ($showExample) {
            echo "  [$word] -> [<not_started>]\n";
        }
        $inputVector = $embedding->embed($word);
        $outputVector = $embedding->embed('<not_started>');
        $data[] = [array_merge([1], $inputVector), $outputVector];
    }

    // Phase 2: Signal end of question, start outputting answer words
    foreach ($answerWords as $word) {
        if ($showExample) {
            echo "  [<end_of_sentence>] -> [$word]\n";
        }
        $inputVector = $embedding->embed('<end_of_sentence>');
        $outputVector = $embedding->embed($word);
        $data[] = [array_merge([1], $inputVector), $outputVector];
    }

    // Reset memory for next sentence
    if ($showExample) {
        echo "  [RESET MEMORY]\n\n";
    }
    $data[] = [];

    $showExample = false;
}

foreach ($data as $datum) {
    if (empty($datum)) {
        echo "[RESET MEMORY]\n";
    } else {
        $inputVectorWithBias = $datum[0];
        $outputVector = $datum[1];

        // Remove bias (first element) from input
        $inputVector = array_slice($inputVectorWithBias, 1);

        $inputWord = $embedding->decode($inputVector);
        $outputWord = $embedding->decode($outputVector);

        echo "[$inputWord] -> [$outputWord]\n";
    }
}

echo "Total training sequences: " . count($data) . "\n\n";

// Fitness function: only score answer generation phase
$fitnessFunction = function (Agent $agent, $dataRow, $otherAgents, $world) use ($embeddingDimensions, $embedding) {
    // Check if expected output is <not_started> (listening phase)
    $inputWithBias = $dataRow[0];
    $inputVector = array_slice($inputWithBias, 1); // Remove bias

    $endOfSentenceVector = $embedding->embed('<end_of_sentence>');
    if ($inputVector !== $endOfSentenceVector) {
        return 0;
    }

    // Otherwise, score the output (answer generation phase)
    $predicted = $agent->getOutputValues();
    $actual = $dataRow[1];

    // Calculate Euclidean distance
    $distance = 0;
    for ($i = 0; $i < $embeddingDimensions; $i++) {
        $diff = $predicted[$i] - $actual[$i];
        $distance += $diff * $diff;
    }
    $distance = sqrt($distance);

    // Return fitness (higher is better, so invert distance)
    return 1.0 / (1.0 + $distance);
};

// Create or load world
$worldName = 'LLM_sequential';
if (file_exists("autosave/world_{$worldName}.txt")) {
    echo "Loading saved world...\n";
    $world = World::loadAutoSaved($worldName);
} else {
    echo "Creating new world...\n";
    $world = new World($worldName);
    $world->createAgents(
        $populationSize,
        $embeddingDimensions + 1,  // +1 for bias
        $embeddingDimensions,
        [],
        true
    );
}

// test
echo "Training for $generations generations...\n";
$world->step($fitnessFunction, $data, $generations, $survivalRate);
echo "\nTraining complete!\n";
echo "Best fitness: " . number_format($world->getBestAgent()->getFitness(), 6) . "\n\n";

$agent = $world->getBestAgent();
$agent->resetMemory();

echo "\n" . str_repeat("=", 80) . "\n";
echo "TRAINING DATA PREDICTIONS\n";
echo str_repeat("=", 80) . "\n";
echo str_pad("INPUT", 20) . " | " . str_pad("EXPECTED", 15) . " | " . str_pad("PREDICTED", 15) . " | MATCH\n";
echo str_repeat("-", 80) . "\n";

foreach ($data as $row) {
    if (empty($row)) {
        echo str_repeat("-", 80) . "\n";
        $agent->resetMemory();
        continue;
    }
    $agent->step($row[0]);

    $actual = $row[1];
    $output = $agent->getOutputValues();

    $inputWord = $embedding->decode(array_slice($row[0], 1));
    $expectedWord = $embedding->decode($actual);
    $match = '';
    if ($inputWord == '<end_of_sentence>') {
        $predictedWord = $embedding->decode($output);
        $match = ($expectedWord === $predictedWord) ? "✓" : "✗";
    } else {
        $predictedWord = '-';
    }

    echo str_pad($inputWord, 20) . " | " .
         str_pad($expectedWord, 15) . " | " .
         str_pad($predictedWord, 15) . " | " .
         $match . "\n";
}

echo str_repeat("=", 80) . "\n";
