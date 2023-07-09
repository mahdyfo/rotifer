<?php

require '../vendor/autoload.php';

/**
 * Options
 *      --verbose: Shows more details
 */

// Crossover probability, 0.5 mean half genes from mother and half from father
const PROBABILITY_CROSSOVER = 0.5;
const PROBABILITY_MUTATE_WEIGHT = 0.05;
const PROBABILITY_MUTATE_ADD_NEURON = 0.05;
const PROBABILITY_MUTATE_REMOVE_NEURON = 0.05;
const SAVE_WORLD_EVERY_GENERATION = 10; // Every x generations, saves world and best agent

use GeneticAutoml\Models\World;

$population = 200;
$data = [
    [[0, 0, 0], [1]],
    [[0, 0, 1], [0.9]],
    [[0, 1, 0], [0.8]],
    [[0, 1, 1], [0.7]],
    [[1, 0, 0], [1]],
    [[1, 0, 1], [0.5]],
    [[1, 1, 0], [0.4]],
    [[1, 1, 1], [0.3]],
];

// Fitness
$fitnessFunction = function (\GeneticAutoml\Models\Agent $agent, $dataRow, $otherAgents) {
    $predictedOutput = $agent->getOutputValues()[0];
    $actualOutput = $dataRow[1][0];

    $connections = count($agent->getGenomeArray());
    return (1.0 - abs($predictedOutput - $actualOutput))/* / (pow($connections, 0.5) == 0 ?: 1)*/;
};

// World
$world = new World();
$world->createAgents($population, 3, 1);
$world->step($fitnessFunction, $data, 300, 0.8);

// Report
var_dump(\GeneticAutoml\Helpers\ReportHelper::agentDetails($world->getBestAgent()));

// Test
$agent = $world->getBestAgent();
foreach ($data as $row) {
    $agent->step($row[0]);
    var_dump(round($agent->getOutputValues()[0], 1) . ' - ' . $agent->getOutputValues()[0]);
}
