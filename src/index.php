<?php

require '../vendor/autoload.php';

use GeneticAutoml\Models\World;

$population = 100;
$data = [
    [[0, 0], [0]],
    [[0, 1], [1]],
    [[1, 0], [1]],
    [[1, 1], [0]],
];

$world = new World();
$world->createAgents($population, 2, 1);
/*$agent1 = new \GeneticAutoml\Models\Agent();
$agent1->connectNeurons($agent1->findOrCreateNeuron(0, 0), $agent1->findOrCreateNeuron(1, 0), 1.32);
$agent1->connectNeurons($agent1->findOrCreateNeuron(0, 1), $agent1->findOrCreateNeuron(1, 0), 1.25);
$agent1->connectNeurons($agent1->findOrCreateNeuron(1, 0), $agent1->findOrCreateNeuron(2, 0), 2.45);

$agent2 = new \GeneticAutoml\Models\Agent();
$agent2->connectNeurons($agent2->findOrCreateNeuron(0, 0), $agent2->findOrCreateNeuron(2, 0), 3.56);
$agent2->connectNeurons($agent2->findOrCreateNeuron(0, 0), $agent2->findOrCreateNeuron(1, 0), 4.84);
$agent2->connectNeurons($agent2->findOrCreateNeuron(1, 0), $agent2->findOrCreateNeuron(2, 0), 5.55);*/

$fitnessFunction = function (\GeneticAutoml\Models\Agent $agent, $dataRow, $otherAgents) {
    $predictedOutput = round($agent->getOutputValues()[0]);
    $actualOutput = $dataRow[1][0];

    if ($predictedOutput - $actualOutput == 0) {
        return 1000;
    }
    return 1/pow($predictedOutput - $actualOutput, 2);
};

$world->step($fitnessFunction, $data, 100, 0.8);
var_dump($world->getBestAgent()->getFitness(), $world->getAgents()[0]->getGenomeString(\GeneticAutoml\Encoders\BinaryEncoder::getInstance()), $world->getAgents()[0]->getGenomeString(\GeneticAutoml\Encoders\HumanEncoder::getInstance()));

// test
$agent = $world->getBestAgent();
$agent->step([0,0]);
var_dump($agent->getOutputValues()[0]);
$agent->step([0,1]);
var_dump($agent->getOutputValues()[0]);
$agent->step([1,0]);
var_dump($agent->getOutputValues()[0]);
$agent->step([1,1]);
var_dump($agent->getOutputValues()[0]);
