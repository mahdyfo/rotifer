<?php

require '../vendor/autoload.php';

/**
 * Options
 *      --verbose: Shows more details
 */

use GeneticAutoml\Models\World;

$population = 100;
$data = [
    [[0, 0, 0], [1]],
    [[0, 0, 1], [0]],
    [[0, 1, 0], [0]],
    [[0, 1, 1], [1]],
    [[1, 0, 0], [1]],
    [[1, 0, 1], [0]],
    [[1, 1, 0], [0]],
    [[1, 1, 1], [1]],
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
    $predictedOutput = $agent->getOutputValues()[0];
    $actualOutput = $dataRow[1][0];

    $hiddenCount = count($agent->getNeuronsByType(\GeneticAutoml\Models\Neuron::TYPE_HIDDEN));
    return (1.0 - abs($predictedOutput - $actualOutput)) / ($hiddenCount + 200);
};

$world->step($fitnessFunction, $data, 200, 0.8);
var_dump($world->getBestAgent()->getFitness(), count($world->getBestAgent()->getNeuronsByType(\GeneticAutoml\Models\Neuron::TYPE_HIDDEN)), $world->getBestAgent()->getGenomeString(\GeneticAutoml\Encoders\HumanEncoder::getInstance()));

// test
$agent = $world->getBestAgent();
foreach ($data as $row) {
    $agent->step($row[0]);
    var_dump(round($agent->getOutputValues()[0]) . ' - ' . $agent->getOutputValues()[0]);
}
