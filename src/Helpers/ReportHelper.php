<?php

namespace Rotifer\Helpers;

use Rotifer\Models\Agent;
use Rotifer\Models\Neuron;

class ReportHelper
{
    public static function agentDetails(Agent $agent): array
    {
        return [
            'fitness' => $agent->getFitness(),
            'input_count' => count($agent->getNeuronsByType(Neuron::TYPE_INPUT)),
            'hidden_neurons_count' => count($agent->getNeuronsByType(Neuron::TYPE_HIDDEN)),
            'output_count' => count($agent->getNeuronsByType(Neuron::TYPE_OUTPUT)),
            'connections_count' => count($agent->getGenomeArray()),
        ];
    }
}