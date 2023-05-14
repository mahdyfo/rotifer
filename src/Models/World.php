<?php

namespace GeneticAutoml\Models;

use Exception;
use GeneticAutoml\Encoders\BinaryEncoder;
use GeneticAutoml\Encoders\Encoder;
use GeneticAutoml\Helpers\ReproductionHelper;

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
    public function createAgents(int $count, $inputNeuronsCount, $outputNeuronsCount, $hasMemory = false): self
    {
        $agents = [];

        for ($i = 0; $i < $count; $i++) {
            $agents[] = (new Agent())
                ->createNeuron(Neuron::TYPE_INPUT, $inputNeuronsCount)
                ->createNeuron(Neuron::TYPE_OUTPUT, $outputNeuronsCount)
                ->setHasMemory($hasMemory)
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

    public function reproduce(Agent $agent1, Agent $agent2): Agent
    {
        $genome1 = $agent1->getGenomeArray();
        $genome2 = $agent2->getGenomeArray();

        // Crossover
        [$genome1, $genome2] = ReproductionHelper::crossover($genome1, $genome2);

        // Dominance
        $childGenome = ReproductionHelper::dominance($genome1, $genome2);

        // Mutation
        $childGenome = ReproductionHelper::mutate($childGenome);

        // Translocation
        $childGenome = ReproductionHelper::translocation($childGenome);

        return Agent::createFromGenome($childGenome);
    }

    /**
     * Run each agent with the data. The count of training rows is the age of the agent
     * @param mixed $fitnessFunction Your custom function to calculate the fitness value for each agent
     * @param array $data [ [[input1, input2, input3], [output1, output2]], [[inputX, inputY, inputZ], [outputA, outputB]] ]
     * @throws Exception
     */
    //TODO: add synchronous param: True to run every agent simultaneously with others, False to age 1 agent completely before running the next
    public function nextGeneration($fitnessFunction, array $data)
    {
        // Run all agents
        $fitnessByAgentKey = [];
        foreach ($this->agents as $key => $agent) {
            // Each data
            $fitness = 0;
            foreach ($data as $row) {
                // Step
                $this->agents[$key]->step($row[0]);

                // Feed data into fitness function
                $otherAgents = $this->getAgents();
                unset($otherAgents[$key]);
                $fitness += $fitnessFunction($this->agents[$key], $row, $otherAgents);
            }
            $this->agents[$key]->setFitness($fitness);
            $fitnessByAgentKey[] = [$fitness, $key];
        }

        // Sort agent keys by their fitness values
        usort($fitnessByAgentKey, function($a, $b) {
            return $b[0] > $a[0] ? 1 : -1;
        });

        return $fitnessByAgentKey;
        // TODO: Reproduce
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
