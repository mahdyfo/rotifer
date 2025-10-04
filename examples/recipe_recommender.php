<?php
/**
 * Recipe Recommendation System
 *
 * This example shows how to recommend recipes based on available ingredients.
 * The system learns which ingredient combinations make good recipes and gives a rating (0-10).
 *
 * Think of it like a chef learning what ingredients go well together!
 */

require __DIR__ . '/../vendor/autoload.php';

use Rotifer\Models\{Agent, World};
use Rotifer\Activations\Activation;

// Configuration
const PROBABILITY_CROSSOVER = 0.5;
const PROBABILITY_MUTATE_WEIGHT = 0.4;
const MUTATE_WEIGHT_COUNT = 1;
const PROBABILITY_MUTATE_ADD_NEURON = 0.04;
const PROBABILITY_MUTATE_REMOVE_NEURON = 0.04;
const PROBABILITY_MUTATE_ADD_GENE = 0.1;
const PROBABILITY_MUTATE_REMOVE_GENE = 0.1;
const ACTIVATION = [Activation::class, 'sigmoid'];
const SAVE_WORLD_EVERY_GENERATION = 0;
const CALCULATE_STEP_TIME = false;
const ONLY_CALCULATE_FIRST_STEP_TIME = false;

// Ingredients (1 = have it, 0 = don't have it)
// Order: Bias, Tomato, Cheese, Pasta, Chicken, Rice, Soy Sauce, Ginger, Beef, Bread, Lettuce
$trainingData = [
    // Italian dishes (high rating)
    [[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], [0.9]],  // Tomato + Cheese + Pasta = Pasta with tomato sauce
    [[1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0], [0.85]], // Tomato + Cheese + Chicken = Chicken Parmesan
    [[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], [0.95]], // All Italian ingredients = Amazing pasta

    // Asian dishes (high rating)
    [[1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0], [0.9]],  // Chicken + Rice + Soy + Ginger = Fried rice
    [[1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0], [0.85]], // Beef + Rice + Soy + Ginger = Beef stir-fry
    [[1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0], [0.8]],  // Chicken + Rice + Soy = Simple fried rice

    // Sandwiches (medium-high rating)
    [[1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1], [0.8]],  // Chicken + Cheese + Bread + Lettuce = Chicken sandwich
    [[1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1], [0.75]], // Beef + Tomato + Cheese + Bread + Lettuce = Burger
    [[1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0], [0.6]],  // Just Cheese + Bread = Simple cheese sandwich

    // Bad combinations (low rating)
    [[1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0], [0.2]],  // Tomato + Pasta + Rice = Weird mix
    [[1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0], [0.15]], // Cheese + Soy + Ginger + Bread = Doesn't work
    [[1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0], [0.1]],  // Pasta + Soy + Beef = Strange combo
    [[1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1], [0.2]],  // Tomato + Rice + Ginger + Lettuce = Odd

    // Missing key ingredients (low-medium rating)
    [[1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0.4]],  // Pasta + Tomato but no protein
    [[1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0], [0.5]],  // Chicken + Rice but no seasoning
    [[1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0], [0.45]], // Just Beef + Bread

    // Single ingredients (very low rating)
    [[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0.1]],  // Just Tomato
    [[1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0.1]],  // Just Pasta
    [[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0.1]],  // Just Rice
];

// Fitness function: How close is the rating to what people actually like?
$fitnessFunction = function (Agent $agent, $dataRow) {
    $predictedRating = $agent->getOutputValues()[0];
    $actualRating = $dataRow[1][0];

    // Closer to actual rating = higher fitness
    $error = abs($predictedRating - $actualRating);
    return 1.0 - $error;
};

echo "Recipe Recommendation System - Training started" . PHP_EOL;
echo "Teaching agents which ingredients go well together..." . PHP_EOL . PHP_EOL;

// Create world with agents
$world = new World('recipe_recommender');
$world->createAgents(
    100,    // 100 agents competing to learn best recipes
    11,     // 11 inputs: bias + 10 ingredients
    1,      // 1 output: recipe rating (0-1 scale, multiply by 10 for display)
    [8],    // Static architecture with 8 hidden neurons (like having 8 taste testers)
    false   // No memory needed
);

// Train for 100 generations
$world->step($fitnessFunction, $trainingData, 100, 0.2);

echo PHP_EOL . "Training complete!" . PHP_EOL . PHP_EOL;

// Test the best agent
$bestAgent = $world->getBestAgent();
echo "Testing recipe recommendations:" . PHP_EOL;
echo str_repeat('-', 80) . PHP_EOL;

$ingredientNames = ['Bias', 'Tomato', 'Cheese', 'Pasta', 'Chicken', 'Rice', 'Soy Sauce', 'Ginger', 'Beef', 'Bread', 'Lettuce'];

$testCases = [
    [[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], "Italian feast: Tomato, Cheese, Pasta, Chicken"],
    [[1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0], "Asian stir-fry: Chicken, Rice, Soy Sauce, Ginger"],
    [[1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1], "Classic burger: Beef, Tomato, Cheese, Bread, Lettuce"],
    [[1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0], "Weird mix: Tomato, Pasta, Rice"],
    [[1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0], "Strange combo: Cheese, Soy Sauce, Bread"],
    [[1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], "Simple salad: Bread, Lettuce"],
];

foreach ($testCases as $test) {
    $bestAgent->reset();
    $bestAgent->step($test[0]);
    $rating = $bestAgent->getOutputValues()[0] * 10; // Convert to 0-10 scale

    // Show which ingredients are used
    $ingredients = [];
    for ($i = 1; $i < count($test[0]); $i++) {
        if ($test[0][$i] == 1) {
            $ingredients[] = $ingredientNames[$i];
        }
    }

    echo $test[1] . PHP_EOL;
    echo "Rating: " . number_format($rating, 1) . "/10";

    if ($rating >= 8) {
        echo " ⭐ Excellent!";
    } elseif ($rating >= 6) {
        echo " 👍 Good";
    } elseif ($rating >= 4) {
        echo " 😐 Okay";
    } else {
        echo " ❌ Not recommended";
    }

    echo PHP_EOL . PHP_EOL;
}

echo str_repeat('-', 80) . PHP_EOL;
echo "The system learned which ingredients combine well together!" . PHP_EOL;
echo "Italian ingredients + Italian = High rating" . PHP_EOL;
echo "Asian ingredients + Asian = High rating" . PHP_EOL;
echo "Mixing different cuisines = Low rating" . PHP_EOL;
