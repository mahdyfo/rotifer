<?php
/**
 * House Price Estimator
 *
 * This example estimates house prices based on features like:
 * - Number of bedrooms
 * - Number of bathrooms
 * - Square footage
 * - Age of the house
 * - Distance to city center
 * - Has garage?
 * - Has garden?
 *
 * Think of it like a real estate agent learning what makes houses valuable!
 */

require __DIR__ . '/../vendor/autoload.php';

use Rotifer\Models\{Agent, World};
use Rotifer\Activations\Activation;

// Configuration
const PROBABILITY_CROSSOVER = 0.5;
const PROBABILITY_MUTATE_WEIGHT = 0.35;
const MUTATE_WEIGHT_COUNT = 1;
const PROBABILITY_MUTATE_ADD_NEURON = 0.05;
const PROBABILITY_MUTATE_REMOVE_NEURON = 0.05;
const PROBABILITY_MUTATE_ADD_GENE = 0.1;
const PROBABILITY_MUTATE_REMOVE_GENE = 0.1;
const ACTIVATION = [Activation::class, 'relu'];
const SAVE_WORLD_EVERY_GENERATION = 0;
const CALCULATE_STEP_TIME = false;
const ONLY_CALCULATE_FIRST_STEP_TIME = false;

// Training data
// [Bias, Bedrooms, Bathrooms, Sq.Ft.(normalized), Age(years normalized), Distance to city(normalized), Garage(0/1), Garden(0/1)]
// -> [Price in $100k normalized]

$trainingData = [
    // Luxury houses (high price)
    [[1, 5/5, 4/4, 3000/3000, (50-5)/50, (20-2)/20, 1, 1], [800/1000]],   // 5BR, 4BA, 3000sqft, 5yr, 2km, garage, garden = $800k
    [[1, 4/5, 3/4, 2500/3000, (50-3)/50, (20-3)/20, 1, 1], [650/1000]],   // 4BR, 3BA, 2500sqft, 3yr, 3km = $650k
    [[1, 5/5, 4/4, 3000/3000, (50-2)/50, (20-5)/20, 1, 1], [750/1000]],   // 5BR, 4BA, 3000sqft, 2yr, 5km = $750k
    [[1, 4/5, 3/4, 2800/3000, (50-4)/50, (20-4)/20, 1, 1], [700/1000]],   // Large, new, close

    // Mid-range houses
    [[1, 3/5, 2/4, 1800/3000, (50-10)/50, (20-8)/20, 1, 1], [450/1000]],  // 3BR, 2BA, 1800sqft, 10yr, 8km = $450k
    [[1, 3/5, 2/4, 1600/3000, (50-8)/50, (20-10)/20, 1, 0], [380/1000]],  // 3BR, 2BA, 1600sqft, 8yr, 10km, no garden = $380k
    [[1, 3/5, 2/4, 2000/3000, (50-12)/50, (20-7)/20, 1, 1], [420/1000]],  // 3BR, 2BA, 2000sqft, 12yr = $420k
    [[1, 3/5, 1/4, 1500/3000, (50-15)/50, (20-12)/20, 0, 1], [320/1000]], // 3BR, 1BA, 1500sqft, older, far = $320k

    // Starter homes (lower price)
    [[1, 2/5, 1/4, 1200/3000, (50-20)/50, (20-15)/20, 0, 0], [250/1000]], // 2BR, 1BA, 1200sqft, 20yr, 15km = $250k
    [[1, 2/5, 1/4, 1000/3000, (50-25)/50, (20-18)/20, 0, 0], [200/1000]], // 2BR, 1BA, 1000sqft, 25yr, 18km = $200k
    [[1, 1/5, 1/4, 800/3000, (50-30)/50, (20-20)/20, 0, 0], [150/1000]],  // 1BR, 1BA, 800sqft, 30yr, far = $150k
    [[1, 2/5, 1/4, 1100/3000, (50-22)/50, (20-16)/20, 0, 1], [230/1000]], // 2BR, small, old but has garden = $230k

    // Small but good location
    [[1, 2/5, 1/4, 1000/3000, (50-10)/50, (20-2)/20, 0, 0], [350/1000]],  // Small but very close to city = $350k
    [[1, 2/5, 2/4, 1200/3000, (50-5)/50, (20-3)/20, 1, 0], [400/1000]],   // Small, new, close, garage = $400k

    // Large but old or far
    [[1, 4/5, 2/4, 2200/3000, (50-35)/50, (20-18)/20, 0, 1], [300/1000]], // Big but old and far = $300k
    [[1, 4/5, 3/4, 2500/3000, (50-40)/50, (20-19)/20, 1, 1], [350/1000]], // Large, very old, far = $350k

    // Premium apartments
    [[1, 3/5, 2/4, 1400/3000, (50-1)/50, (20-1)/20, 1, 0], [550/1000]],   // New condo, city center = $550k
    [[1, 2/5, 2/4, 1100/3000, (50-2)/50, (20-1)/20, 1, 0], [480/1000]],   // Small but prime location = $480k

    // Fixer-uppers
    [[1, 3/5, 1/4, 1500/3000, (50-45)/50, (20-10)/20, 0, 1], [180/1000]], // Old, needs work = $180k
    [[1, 3/5, 2/4, 1800/3000, (50-42)/50, (20-14)/20, 0, 0], [210/1000]], // Old, average = $210k
];

// Fitness function: How accurate is the price estimate?
$fitnessFunction = function (Agent $agent, $dataRow) {
    $estimatedPrice = $agent->getOutputValues()[0];
    $actualPrice = $dataRow[1][0];

    // Closer estimate = higher fitness
    $error = abs($estimatedPrice - $actualPrice);
    return 1.0 - $error;
};

echo "House Price Estimator - Training started" . PHP_EOL;
echo "Learning what makes houses valuable..." . PHP_EOL . PHP_EOL;

// Create world with agents
$world = new World('house_price_estimator');
$world->createAgents(
    100,    // 100 agents learning pricing patterns
    8,      // 8 inputs: bias + 7 house features
    1,      // 1 output: estimated price
    [10, 6], // Static architecture: 10 neurons in first layer, 6 in second (like having multiple appraisers)
    false   // No memory needed
);

// Train for 150 generations
$world->step($fitnessFunction, $trainingData, 150, 0.2);

echo PHP_EOL . "Training complete!" . PHP_EOL . PHP_EOL;

// Test the best agent
$bestAgent = $world->getBestAgent();
echo "Testing price estimates:" . PHP_EOL;
echo str_repeat('-', 80) . PHP_EOL;

$testCases = [
    [[1, 4/5, 3/4, 2400/3000, (50-5)/50, (20-4)/20, 1, 1], "Dream house: 4BR, 3BA, 2400sqft, 5 years old, 4km from city, garage, garden"],
    [[1, 3/5, 2/4, 1700/3000, (50-12)/50, (20-9)/20, 1, 1], "Family home: 3BR, 2BA, 1700sqft, 12 years old, 9km from city, garage, garden"],
    [[1, 2/5, 1/4, 1100/3000, (50-20)/50, (20-15)/20, 0, 0], "Starter home: 2BR, 1BA, 1100sqft, 20 years old, 15km from city"],
    [[1, 5/5, 4/4, 3000/3000, (50-2)/50, (20-2)/20, 1, 1], "Luxury mansion: 5BR, 4BA, 3000sqft, brand new, close to city"],
    [[1, 2/5, 2/4, 1000/3000, (50-3)/50, (20-1)/20, 1, 0], "City apartment: 2BR, 2BA, 1000sqft, new, city center"],
    [[1, 4/5, 2/4, 2000/3000, (50-40)/50, (20-18)/20, 0, 1], "Fixer-upper: 4BR, 2BA, 2000sqft, 40 years old, far from city"],
];

foreach ($testCases as $test) {
    $bestAgent->reset();
    $bestAgent->step($test[0]);
    $price = $bestAgent->getOutputValues()[0] * 1000; // Convert back to thousands

    echo $test[1] . PHP_EOL;
    echo "Estimated Price: $" . number_format($price, 0) . "k";

    // Price category
    if ($price >= 600) {
        echo " 💎 Luxury";
    } elseif ($price >= 400) {
        echo " 🏡 Premium";
    } elseif ($price >= 300) {
        echo " 🏠 Mid-range";
    } else {
        echo " 🔑 Affordable";
    }

    echo PHP_EOL . PHP_EOL;
}

echo str_repeat('-', 80) . PHP_EOL;
echo "What the system learned:" . PHP_EOL;
echo "• More bedrooms & bathrooms = Higher price" . PHP_EOL;
echo "• Larger square footage = Higher price" . PHP_EOL;
echo "• Newer houses = Higher price" . PHP_EOL;
echo "• Closer to city = Higher price" . PHP_EOL;
echo "• Garage & garden = Add value" . PHP_EOL;
echo "• Location can outweigh size (small house downtown > big house far away)" . PHP_EOL;
