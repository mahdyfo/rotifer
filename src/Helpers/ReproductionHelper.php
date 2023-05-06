<?php

namespace GeneticAutoml\Helpers;

use GeneticAutoml\Encoders\BinaryEncoder;
use GeneticAutoml\Models\Agent;

class ReproductionHelper
{
    public static function crossover(Agent $agent1, Agent $agent2): array
    {
        $binaryEncoder = new BinaryEncoder();
        $genomes = [$agent1->getGenomeArray($binaryEncoder), $agent2->getGenomeArray($binaryEncoder)];
        shuffle($genomes);

        $crossPoint = mt_rand(30, 70) / 100;
        $crossPoint = round($crossPoint * count($genomes[0]));

        $part1 = array_slice($genomes[0], 0, $crossPoint);
        $part2 = array_slice($genomes[1], $crossPoint);

        return array_merge($part1, $part2);
    }
}