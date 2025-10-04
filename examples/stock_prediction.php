<?php
/**
 * Stock Price Prediction Example
 *
 * This example shows how to predict tomorrow's stock price based on the past 5 days of prices.
 * The system learns patterns like trends, momentum, and price movements.
 */

require __DIR__ . '/../vendor/autoload.php';

use Rotifer\Models\{Agent, World};
use Rotifer\Activations\Activation;

// Configuration
const PROBABILITY_CROSSOVER = 0.5;
const PROBABILITY_MUTATE_WEIGHT = 0.3;
const MUTATE_WEIGHT_COUNT = 1;
const PROBABILITY_MUTATE_ADD_NEURON = 0.05;
const PROBABILITY_MUTATE_REMOVE_NEURON = 0.05;
const PROBABILITY_MUTATE_ADD_GENE = 0.1;
const PROBABILITY_MUTATE_REMOVE_GENE = 0.1;
const ACTIVATION = [Activation::class, 'tanh'];
const SAVE_WORLD_EVERY_GENERATION = 0;
const CALCULATE_STEP_TIME = false;
const ONLY_CALCULATE_FIRST_STEP_TIME = false;

// Sample stock price data (simplified - normally you'd use real historical data)
// Format: [previous 5 days prices] -> [next day price]
$trainingData = [
    // Upward trend
    [[1, 100, 102, 104, 106, 108], [110]],
    [[1, 102, 104, 106, 108, 110], [112]],
    [[1, 104, 106, 108, 110, 112], [114]],

    // Downward trend
    [[1, 120, 118, 116, 114, 112], [110]],
    [[1, 118, 116, 114, 112, 110], [108]],
    [[1, 116, 114, 112, 110, 108], [106]],

    // Volatile (up and down)
    [[1, 100, 105, 103, 107, 102], [104]],
    [[1, 105, 103, 107, 102, 104], [106]],
    [[1, 103, 107, 102, 104, 106], [103]],

    // Stable prices
    [[1, 100, 100, 101, 100, 101], [100]],
    [[1, 100, 101, 100, 101, 100], [101]],
    [[1, 101, 100, 101, 100, 101], [100]],

    // Sharp rise
    [[1, 100, 105, 110, 115, 120], [125]],
    [[1, 105, 110, 115, 120, 125], [130]],

    // Sharp drop
    [[1, 120, 115, 110, 105, 100], [95]],
    [[1, 115, 110, 105, 100, 95], [90]],
];

// Normalize prices to range 0-1 for better learning
$maxPrice = 150;
foreach ($trainingData as &$row) {
    for ($i = 0; $i < count($row[0]); $i++) {
        if ($i > 0) { // Skip bias
            $row[0][$i] = $row[0][$i] / $maxPrice;
        }
    }
    $row[1][0] = $row[1][0] / $maxPrice;
}

// Fitness function: How close is the prediction to the actual price?
$fitnessFunction = function (Agent $agent, $dataRow) {
    $predictedPrice = $agent->getOutputValues()[0];
    $actualPrice = $dataRow[1][0];

    // Closer prediction = higher fitness
    $error = abs($predictedPrice - $actualPrice);
    return 1.0 - $error;
};

echo "Stock Price Prediction - Training started" . PHP_EOL;
echo "Learning to predict tomorrow's price from the last 5 days..." . PHP_EOL . PHP_EOL;

// Create world with agents
$world = new World('stock_prediction');
$world->createAgents(
    100,    // 100 agents competing to find best prediction strategy
    6,      // 6 inputs: bias + 5 days of prices
    1,      // 1 output: predicted next day price
    [],     // Dynamic architecture (agents evolve their own structure)
    false   // No memory needed (each prediction is independent)
);

// Train for 50 generations
$world->step($fitnessFunction, $trainingData, 50, 0.2);

echo PHP_EOL . "Training complete!" . PHP_EOL . PHP_EOL;

// Test the best agent
$bestAgent = $world->getBestAgent();
echo "Testing predictions:" . PHP_EOL;
echo str_repeat('-', 80) . PHP_EOL;

// Test with some patterns
$testCases = [
    [[1, 100/150, 105/150, 110/150, 115/150, 120/150], "Upward trend: 100->105->110->115->120"],
    [[1, 120/150, 115/150, 110/150, 105/150, 100/150], "Downward trend: 120->115->110->105->100"],
    [[1, 100/150, 100/150, 101/150, 100/150, 101/150], "Stable prices: 100->100->101->100->101"],
];

foreach ($testCases as $test) {
    $bestAgent->reset();
    $bestAgent->step($test[0]);
    $prediction = $bestAgent->getOutputValues()[0] * $maxPrice;

    echo $test[1] . PHP_EOL;
    echo "Predicted next price: $" . number_format($prediction, 2) . PHP_EOL . PHP_EOL;
}

echo str_repeat('-', 80) . PHP_EOL;
echo "Note: This is a simplified example. Real stock prediction requires much more data," . PHP_EOL;
echo "including volume, market indicators, news sentiment, and many other factors." . PHP_EOL;
