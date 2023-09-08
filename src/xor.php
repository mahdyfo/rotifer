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
const PROBABILITY_MUTATE_WEIGHT = 0.4; // Percentage of agents to get mutated
const MUTATE_WEIGHT_COUNT = 2; // number of weight mutations in every agent
const PROBABILITY_MUTATE_ADD_NEURON = 0;
const PROBABILITY_MUTATE_ADD_GENE = 0;
const PROBABILITY_MUTATE_REMOVE_NEURON = 0;
const PROBABILITY_MUTATE_REMOVE_GENE = 0;
const ACTIVATION = [Activation::class, 'sigmoid'];
const SAVE_WORLD_EVERY_GENERATION = 200;
const CALCULATE_STEP_TIME = false;
const ONLY_CALCULATE_FIRST_STEP_TIME = false;

$population = 100;
$generations = 30;
$layers = [3, 2];

$data = [
    [[1, 0, 0], [0]],
    [[1, 0, 1], [1]],
    [[1, 1, 0], [1]],
    [[1, 1, 1], [0]],
    //^ bias
];

// Fitness
$fitnessFunction = function (\GeneticAutoml\Models\Agent $agent, $dataRow, $otherAgents, $world) {
    $predictedOutput = $agent->getOutputValues()[0];
    $actualOutput = $dataRow[1][0];

    return (1.0 - abs($predictedOutput - $actualOutput));
};

// World
$world = new World();
$world->createAgents($population, count($data[0][0]), 1, $layers, false);
$world->step($fitnessFunction, $data, $generations, 0.8);

// Test/Predict
/** @var \GeneticAutoml\Models\StaticAgent $agent */
$agent = $world->getBestAgent();
$agent->resetValues();
foreach ($data as $row) {
    $agent->step($row[0]);
    var_dump('Input: ' . implode(',', $row[0]) . ' - Round: ' . round($agent->getOutputValues()[0]) . ' - Raw: ' . $agent->getOutputValues()[0]);
}
