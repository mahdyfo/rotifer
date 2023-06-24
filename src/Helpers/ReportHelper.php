<?php

namespace GeneticAutoml\Helpers;

use GeneticAutoml\Models\Agent;
use GeneticAutoml\Models\Neuron;

class ReportHelper
{
    public static function agentDetails(Agent $agent): array
    {
        return [
            'fitness' => $agent->getFitness(),
            'hidden_neurons_count' => count($agent->getNeuronsByType(Neuron::TYPE_HIDDEN)),
            'connections_count' => count($agent->getGenomeArray()),
        ];
    }
}