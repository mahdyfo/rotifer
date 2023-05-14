<?php

require '../vendor/autoload.php';

use GeneticAutoml\Models\World;

$population = 3;
$data = [
    [[1, 0, 0], [1]]
];

$world = new World();
$world->createAgents($population, 3, 1);
/*$agent1 = new \GeneticAutoml\Models\Agent();
$agent1->connectNeurons($agent1->findOrCreateNeuron(0, 0), $agent1->findOrCreateNeuron(1, 0), 1.32);
$agent1->connectNeurons($agent1->findOrCreateNeuron(0, 1), $agent1->findOrCreateNeuron(1, 0), 1.25);
$agent1->connectNeurons($agent1->findOrCreateNeuron(1, 0), $agent1->findOrCreateNeuron(2, 0), 2.45);

$agent2 = new \GeneticAutoml\Models\Agent();
$agent2->connectNeurons($agent2->findOrCreateNeuron(0, 0), $agent2->findOrCreateNeuron(2, 0), 3.56);
$agent2->connectNeurons($agent2->findOrCreateNeuron(0, 0), $agent2->findOrCreateNeuron(1, 0), 4.84);
$agent2->connectNeurons($agent2->findOrCreateNeuron(1, 0), $agent2->findOrCreateNeuron(2, 0), 5.55);*/

$fitnessFunction = function (\GeneticAutoml\Models\Agent $agent, $dataRow, $otherAgents) {
    return $agent->getOutputValues()[0] * 2;
};

var_dump($world->nextGeneration($fitnessFunction, $data));
