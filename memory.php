<?php

require 'vendor/autoload.php';

use Rotifer\Activations\Activation;
use Rotifer\Models\Agent;
use Rotifer\Models\World;

/**
 * Options
 *      --quiet: Hide details
 */

const PROBABILITY_CROSSOVER = 0.5; // Crossover probability, 0.5 mean half genes from mother and half from father
const PROBABILITY_MUTATE_WEIGHT = 0.4; // Percentage of agents to get mutated
const MUTATE_WEIGHT_COUNT = 1; // number of weight mutations in every agent
const PROBABILITY_MUTATE_ADD_NEURON = 0.04;
const PROBABILITY_MUTATE_REMOVE_NEURON = 0.04;
const PROBABILITY_MUTATE_ADD_GENE = 0.1;
const PROBABILITY_MUTATE_REMOVE_GENE = 0.1;
const ACTIVATION = [Activation::class, 'sigmoid'];
const SAVE_WORLD_EVERY_GENERATION = 10; // 0: don't save
const CALCULATE_STEP_TIME = false;
const ONLY_CALCULATE_FIRST_STEP_TIME = false;

$population = 100;
$generations = 80;

$data = [
    [[1], [0]],
    [[1], [1]],
    [[1], [0]],
    [[1], [1]],
    [[1], [0]],
    [[1], [1]],
];

// Fitness
$fitnessFunction = function (Agent $agent, $dataRow, $otherAgents, $world) {
    $predictedOutput = $agent->getOutputValues()[0];
    $actualOutput = $dataRow[1][0];
    return (1.0 - abs(abs($predictedOutput) - abs($actualOutput)));
};

// World
$name = 'memory';
echo 'Max possible score: ' . count($data) . "\n\n";
$world = new World($name);

// Train
$world->createAgents($population, count($data[0][0]), count($data[0][1]), [], true);
//$world = World::loadAutoSaved($name, true);
$world->step($fitnessFunction, $data, $generations, 0.3);

// Test/Predict
$agent = $world->getBestAgent();
$agent->resetMemory();
foreach ($data as $row) {
    $agent->step($row[0]);

    $actual = $row[1][0];
    $output = $agent->getOutputValues()[0];
    var_dump(
        'Input: ' . round($row[0][0], 3) . '   ' .
        str_pad('Actual output: ' . round($actual, 3), 19)
        . 'Predicted: ' . round($output, 3)
    );
}
