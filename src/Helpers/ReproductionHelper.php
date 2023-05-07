<?php

namespace GeneticAutoml\Helpers;

use GeneticAutoml\Encoders\BinaryEncoder;
use GeneticAutoml\Models\Agent;

class ReproductionHelper
{
    public static function crossover(Agent $agent1, Agent $agent2): Agent
    {
        $genomes = [$agent1->getGenomeArray(), $agent2->getGenomeArray()];
        shuffle($genomes);

        $crossPoint = mt_rand(30, 70) / 100;
        $crossPoint = round($crossPoint * count($genomes[0]));

        // Always > 0, at least 1
        if ($crossPoint == 0 && count($genomes[0]) > 1) {
            $crossPoint = 1;
        }

        // Always < max length, max count - 1
        if ($crossPoint == count($genomes[0])) {
            $crossPoint = count($genomes[0]) - 1;
        }

        $part1 = array_slice($genomes[0], 0, $crossPoint);
        $part2 = array_slice($genomes[1], $crossPoint);

        $newGenome = array_merge($part1, $part2);

        return Agent::createFromGenome($newGenome);
    }

    public static function mutate(Agent $agent)
    {

    }
}