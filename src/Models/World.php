<?php

namespace GeneticAutoml\Models;

use Exception;
use GeneticAutoml\Encoders\BinaryEncoder;
use GeneticAutoml\Encoders\Encoder;

class World
{
    /**
     * @var Agent[]
     */
    private array $agents = [];

    /**
     * @param int $count How many agents to create
     * @param $inputNeuronsCount
     * @param $outputNeuronsCount
     * @return World
     * @throws Exception
     */
    public function createAgents(int $count, $inputNeuronsCount, $outputNeuronsCount): self
    {
        $agents = [];

        for ($i = 0; $i < $count; $i++) {
            $agents[] = (new Agent())
                ->createNeuron(Neuron::TYPE_INPUT, $inputNeuronsCount)
                ->createNeuron(Neuron::TYPE_OUTPUT, $outputNeuronsCount)
                ->initRandomConnections();
        }

        return $this->setAgents($agents);
    }

    /**
     * @return Agent[]
     */
    public function getAgents(): array
    {
        return $this->agents;
    }

    public function setAgents($agents): self
    {
        $this->agents = $agents;

        return $this;
    }

    public function getGenomesString(Encoder $encoder = null, $geneIterationCallback = null, $geneSeparator = ';', $genomeSeparator = "\n"): string
    {
        if (is_null($encoder)) {
            $encoder = BinaryEncoder::getInstance();
        }
        $genomes = [];
        foreach ($this->agents as $agent) {
            $genomes[] = $agent->getGenomeString($encoder, $geneSeparator, $geneIterationCallback);
        }
        return implode($genomeSeparator, $genomes);
    }
}
