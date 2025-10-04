<?php
/**
 * Weather Forecasting Example
 *
 * This example predicts tomorrow's weather based on today's conditions.
 * The system learns patterns like: "If pressure is dropping and humidity rising, it might rain"
 *
 * Think of it like learning from experience: "When I see these clouds, it usually rains"
 */

require __DIR__ . '/../vendor/autoload.php';

use Rotifer\Models\{Agent, World};
use Rotifer\Activations\Activation;

// Configuration
const PROBABILITY_CROSSOVER = 0.5;
const PROBABILITY_MUTATE_WEIGHT = 0.3;
const MUTATE_WEIGHT_COUNT = 1;
const PROBABILITY_MUTATE_ADD_NEURON = 0.06;
const PROBABILITY_MUTATE_REMOVE_NEURON = 0.06;
const PROBABILITY_MUTATE_ADD_GENE = 0.12;
const PROBABILITY_MUTATE_REMOVE_GENE = 0.12;
const ACTIVATION = [Activation::class, 'sigmoid'];
const SAVE_WORLD_EVERY_GENERATION = 0;
const CALCULATE_STEP_TIME = false;
const ONLY_CALCULATE_FIRST_STEP_TIME = false;

// Weather inputs (normalized 0-1):
// [Bias, Temperature(0-40°C normalized), Humidity(%), Pressure(normalized), Wind Speed(normalized), Cloud Cover(%)]
// Outputs: [Sunny, Cloudy, Rainy, Stormy] (one will be highest)

$trainingData = [
    // SUNNY weather patterns
    [[1, 0.75, 0.3, 0.8, 0.2, 0.1], [1, 0, 0, 0]],  // Hot, low humidity, high pressure, light wind, clear
    [[1, 0.7, 0.4, 0.75, 0.3, 0.2], [1, 0, 0, 0]],  // Warm, low humidity, high pressure
    [[1, 0.65, 0.35, 0.8, 0.25, 0.15], [1, 0, 0, 0]], // Nice conditions
    [[1, 0.8, 0.25, 0.85, 0.15, 0.05], [1, 0, 0, 0]], // Very hot and clear

    // CLOUDY weather patterns
    [[1, 0.5, 0.6, 0.5, 0.4, 0.7], [0, 1, 0, 0]],   // Moderate temp, higher humidity, medium pressure
    [[1, 0.55, 0.55, 0.55, 0.35, 0.65], [0, 1, 0, 0]], // Overcast conditions
    [[1, 0.45, 0.65, 0.45, 0.45, 0.75], [0, 1, 0, 0]], // Grey day
    [[1, 0.6, 0.5, 0.6, 0.3, 0.6], [0, 1, 0, 0]],   // Partly cloudy

    // RAINY weather patterns
    [[1, 0.4, 0.85, 0.3, 0.5, 0.9], [0, 0, 1, 0]],  // Cool, high humidity, low pressure, cloudy
    [[1, 0.45, 0.9, 0.25, 0.55, 0.95], [0, 0, 1, 0]], // Very wet conditions
    [[1, 0.35, 0.8, 0.35, 0.45, 0.85], [0, 0, 1, 0]], // Rain approaching
    [[1, 0.5, 0.88, 0.28, 0.6, 0.92], [0, 0, 1, 0]], // Steady rain
    [[1, 0.42, 0.95, 0.2, 0.48, 0.98], [0, 0, 1, 0]], // Heavy rain

    // STORMY weather patterns
    [[1, 0.3, 0.95, 0.15, 0.9, 1.0], [0, 0, 0, 1]],  // Cold, very humid, very low pressure, strong wind
    [[1, 0.35, 0.98, 0.1, 0.95, 0.95], [0, 0, 0, 1]], // Severe conditions
    [[1, 0.25, 0.9, 0.18, 0.85, 0.98], [0, 0, 0, 1]], // Storm brewing
    [[1, 0.4, 0.92, 0.12, 0.88, 1.0], [0, 0, 0, 1]], // Thunderstorm

    // Transition patterns (learning edge cases)
    [[1, 0.6, 0.5, 0.7, 0.35, 0.4], [0, 1, 0, 0]],  // Sunny becoming cloudy
    [[1, 0.5, 0.7, 0.4, 0.5, 0.8], [0, 0, 1, 0]],   // Cloudy becoming rainy
    [[1, 0.55, 0.45, 0.65, 0.25, 0.3], [1, 0, 0, 0]], // Cloudy clearing up
];

// Fitness function: How well does it predict the weather?
$fitnessFunction = function (Agent $agent, $dataRow) {
    $predictions = $agent->getOutputValues();
    $actual = $dataRow[1];

    // Calculate how close predictions are to actual
    $totalError = 0;
    for ($i = 0; $i < 4; $i++) {
        $totalError += abs($predictions[$i] - $actual[$i]);
    }

    // Lower error = higher fitness
    return 1.0 - ($totalError / 4);
};

echo "Weather Forecasting System - Training started" . PHP_EOL;
echo "Learning to predict weather from atmospheric conditions..." . PHP_EOL . PHP_EOL;

// Create world with agents
$world = new World('weather_forecast');
$world->createAgents(
    120,    // 120 agents competing to find best weather patterns
    6,      // 6 inputs: bias + temperature, humidity, pressure, wind speed, cloud cover
    4,      // 4 outputs: sunny, cloudy, rainy, stormy (highest value wins)
    [],     // Dynamic architecture
    false   // No memory needed
);

// Train for 80 generations
$world->step($fitnessFunction, $trainingData, 80, 0.25);

echo PHP_EOL . "Training complete!" . PHP_EOL . PHP_EOL;

// Test the best agent
$bestAgent = $world->getBestAgent();
echo "Testing weather predictions:" . PHP_EOL;
echo str_repeat('-', 80) . PHP_EOL;

$weatherTypes = ['☀️ Sunny', '☁️ Cloudy', '🌧️ Rainy', '⛈️ Stormy'];

$testCases = [
    [[1, 0.75, 0.3, 0.8, 0.2, 0.1], "Perfect summer day (hot, dry, high pressure, calm)"],
    [[1, 0.5, 0.6, 0.5, 0.4, 0.7], "Overcast conditions (moderate everything)"],
    [[1, 0.4, 0.85, 0.3, 0.5, 0.9], "Looks like rain (humid, low pressure, cloudy)"],
    [[1, 0.3, 0.95, 0.15, 0.9, 1.0], "Storm warning! (very humid, very low pressure, strong wind)"],
    [[1, 0.65, 0.4, 0.7, 0.3, 0.3], "Nice spring day (warm, comfortable)"],
    [[1, 0.45, 0.9, 0.25, 0.55, 0.95], "Heavy rain conditions (very wet, low pressure)"],
];

foreach ($testCases as $test) {
    $bestAgent->reset();
    $bestAgent->step($test[0]);
    $predictions = $bestAgent->getOutputValues();

    // Find which weather type has highest prediction
    $maxIndex = 0;
    $maxValue = $predictions[0];
    for ($i = 1; $i < 4; $i++) {
        if ($predictions[$i] > $maxValue) {
            $maxValue = $predictions[$i];
            $maxIndex = $i;
        }
    }

    $confidence = $maxValue * 100;

    echo "Conditions: " . $test[1] . PHP_EOL;
    echo "Prediction: " . $weatherTypes[$maxIndex] . " (Confidence: " . number_format($confidence, 1) . "%)" . PHP_EOL;

    // Show all probabilities
    echo "  Breakdown: ";
    for ($i = 0; $i < 4; $i++) {
        echo $weatherTypes[$i] . " " . number_format($predictions[$i] * 100, 1) . "%";
        if ($i < 3) echo ", ";
    }
    echo PHP_EOL . PHP_EOL;
}

echo str_repeat('-', 80) . PHP_EOL;
echo "The system learned weather patterns:" . PHP_EOL;
echo "• High pressure + low humidity + clear skies = Sunny" . PHP_EOL;
echo "• Medium conditions = Cloudy" . PHP_EOL;
echo "• Low pressure + high humidity + clouds = Rainy" . PHP_EOL;
echo "• Very low pressure + very high humidity + strong wind = Stormy" . PHP_EOL;
