<?php

require '../vendor/autoload.php';

use GeneticAutoml\Models\World;

$population = 5;
$inputs = [1, 0, 0];
$outputs = [1];

$world = new World();
$world->createAgents($population, count($inputs), count($outputs));

$child = \GeneticAutoml\Helpers\ReproductionHelper::crossover($world->getAgents()[0], $world->getAgents()[1]);
var_dump($child);
