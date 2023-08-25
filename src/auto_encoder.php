<?php

require '../vendor/autoload.php';
use GeneticAutoml\Activations\Activation;
use GeneticAutoml\Models\StaticAgent;
use GeneticAutoml\Models\World;
/**
 * Options
 *      --quiet: Hide details
 */

// Crossover probability, 0.5 mean half genes from mother and half from father
const PROBABILITY_CROSSOVER = 0.5;
const PROBABILITY_MUTATE_WEIGHT = 0.4;
const MUTATE_WEIGHT_COUNT = 1; // number of weight mutations in every agent
const PROBABILITY_MUTATE_ADD_NEURON = 0;
const PROBABILITY_MUTATE_ADD_GENE = 0;
const PROBABILITY_MUTATE_REMOVE_NEURON = 0;
const PROBABILITY_MUTATE_REMOVE_GENE = 0;
const ACTIVATION = [Activation::class, 'sigmoid'];
const SAVE_WORLD_EVERY_GENERATION = 10; // Every x generations, saves world and best agent
const CALCULATE_STEP_TIME = false;
const ONLY_CALCULATE_FIRST_STEP_TIME = false;

$generations = 100;
$population = 100;
$data = [
    [[0.1, 0.2, 0.3, 0.4, 1, 0, 0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4, 1, 0, 0.1, 0.2, 0.3, 0.4]],
    [[0.2, 0.3, 0.4, 0.5, 1, 0, 0.2, 0.3, 0.4, 0.5], [0.2, 0.3, 0.4, 0.5, 1, 0, 0.2, 0.3, 0.4, 0.5]]
];
$layers = [6, 2, 6];

// Fitness
$fitnessFunction = function (StaticAgent $agent, $dataRow, $otherAgents, $world) {
    $predictedOutputs = $agent->getOutputValues();
    $actualOutputs = $dataRow[1];

    $predictionAccuracy = 0;
    for ($i = 0; $i < count($predictedOutputs); $i++) {
        $predictionAccuracy += (1.0 - abs($predictedOutputs[$i] - $actualOutputs[$i]));
    }
    return $predictionAccuracy;
};

// World
print_r('Max possible score: ' . count($data[0][0]) * count($data) . PHP_EOL);
$world = new World('autoencoder_example');
$world = $world->createAgents($population, count($data[0][0]), count($data[0][0]), $layers);
//$world = World::loadAutoSaved($symbol . '_autoencoder');
$world->step($fitnessFunction, $data, $generations, 0.8);

// Report
print_r(\GeneticAutoml\Helpers\ReportHelper::agentDetails($world->getBestAgent()));

// Test
/** @var \GeneticAutoml\Models\StaticAgent $agent */
$agent = $world->getBestAgent();
$agent->resetValues();
$total = count($data);
$correctness = 0;
foreach ($data as $row) {
    $agent->step($row[0]);
    foreach ($row[0] as $key => $input) {
        $actual = $input;
        $output = $agent->getOutputValues()[$key];
        var_dump(
            str_pad('Actual: ' . round($actual, 3), 17)
            . 'Predicted: ' . round($output, 3)
        );
        $correctness += 1 - abs(abs($actual) - abs($output));
    }
}

$totalRecords = count($data) * count($data[0][0]);
print_r('Accuracy: ' . round($correctness / $totalRecords, 2) * 100 . '%') . PHP_EOL;