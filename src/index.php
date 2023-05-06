<?php

require '../vendor/autoload.php';

use GeneticAutoml\Activations\Activation;
use GeneticAutoml\Models\Agent;
use GeneticAutoml\Models\Neuron;

$inputs = [1, 0, 0];
$outputs = [1];

$agent1 = new Agent();
$agent1->setActivation([Activation::class, 'sigmoid'])
    ->createNeuron(Neuron::TYPE_INPUT, 3)
    ->createNeuron(Neuron::TYPE_OUTPUT)
    ->initRandomConnections();

$agent2 = new Agent();
$agent2->setActivation([Activation::class, 'sigmoid'])
    ->createNeuron(Neuron::TYPE_INPUT, 3)
    ->createNeuron(Neuron::TYPE_OUTPUT)
    ->initRandomConnections();

$agent1->reproduce($agent2);

//$encoder = new BinaryEncoder();
//echo $genome = $agent->getGenomeString($encoder, "\n");
