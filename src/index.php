<?php

require '../vendor/autoload.php';
use GeneticAutoml\Activations\Activation;
use GeneticAutoml\Models\World;

/**
 * Options
 *      --quiet: Hide details
 */

// Crossover probability, 0.5 mean half genes from mother and half from father
const PROBABILITY_CROSSOVER = 0.5;
const PROBABILITY_MUTATE_WEIGHT = 0.1;
const MUTATE_WEIGHT_COUNT = 1; // number of weight mutations in every agent
const PROBABILITY_MUTATE_ADD_NEURON = 0.05;
const PROBABILITY_MUTATE_ADD_GENE = 0.1;
const PROBABILITY_MUTATE_REMOVE_NEURON = 0.05;
const PROBABILITY_MUTATE_REMOVE_GENE = 0.1;
const ACTIVATION = [Activation::class, 'sigmoid'];
const SAVE_WORLD_EVERY_GENERATION = 50; // Every x generations, saves world and best agent
const CALCULATE_STEP_TIME = true;
const ONLY_CALCULATE_FIRST_STEP_TIME = true;

$population = 200;
$data = [
    [[1, 0, 0], [0]],
    [[1, 0, 0], [1]],
    [[1, 0, 0], [1]],
    [[1, 0, 0], [0]],
];

// Fitness
$fitnessFunction = function (\GeneticAutoml\Models\Agent $agent, $dataRow, $otherAgents, $world) {
    $predictedOutput = $agent->getOutputValues()[0];
    $actualOutput = $dataRow[1][0];

    $connections = count($agent->getGenomeArray());
    return (1.0 - abs($predictedOutput - $actualOutput)) / pow($connections, 0.1);
};

// World
$world = new World();
$world->createAgents($population, 3, 1, [], true);
$world->step($fitnessFunction, $data, 100, 0.8);

// Report
var_dump(\GeneticAutoml\Helpers\ReportHelper::agentDetails($world->getBestAgent()));

// Test
$agent = $world->getBestAgent();
$agent->resetValues();
foreach ($data as $row) {
    $agent->step($row[0]);
    var_dump('Input: ' . implode(',', $row[0]) . ' - Round: ' . round($agent->getOutputValues()[0]) . ' - Raw: ' . $agent->getOutputValues()[0]);
}
