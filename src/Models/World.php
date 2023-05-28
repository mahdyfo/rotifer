<?php

namespace GeneticAutoml\Models;

use Exception;
use GeneticAutoml\Encoders\BinaryEncoder;
use GeneticAutoml\Encoders\Encoder;
use GeneticAutoml\Encoders\HumanEncoder;
use GeneticAutoml\Helpers\ReproductionHelper;

class World
{
    /**
     * @var Agent[]
     */
    private array $agents = [];

    private ?Agent $bestAgent = null;

    /**
     * @param int $count How many agents to create
     * @param $inputNeuronsCount
     * @param $outputNeuronsCount
     * @param bool $hasMemory Do they have memory or they should be trained for each training row without any previous knowledge
     * @return World
     * @throws Exception
     */
    public function createAgents(int $count, $inputNeuronsCount, $outputNeuronsCount, bool $hasMemory = false): self
    {
        if ($count <= 1) {
            throw new Exception('The world cannot have only 1 agent.');
        }
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

    /**
     * @param Agent $agent1
     * @param Agent $agent2
     * @return Agent
     * @throws Exception
     */
    public function reproduce(Agent $agent1, Agent $agent2): Agent
    {
        $genome1 = $agent1->getGenomeArray();
        $genome2 = $agent2->getGenomeArray();

        // Do the reproduction process and if there is no gene, repeat the reproduction
        $attempts = 0;
        do {
            // Crossover
            [$newGenome1, $newGenome2] = ReproductionHelper::crossover($genome1, $genome2);

            // Dominance
            $newGenome = ReproductionHelper::dominance($newGenome1, $newGenome2);

            // Mutation
            $newGenome = ReproductionHelper::mutate($newGenome);

            // Translocation
            $newGenome = ReproductionHelper::translocation($newGenome);

            $agent = Agent::createFromGenome($newGenome);

            $attempts++;
            if ($attempts > 100) {
                throw new Exception('Tried ' . $attempts
                    . ' times to reproduce but it always generated agent with no genes. genome1: '
                    . $agent1->getGenomeString(HumanEncoder::getInstance())
                    . ' genome2: ' . $agent2->getGenomeString(HumanEncoder::getInstance())
                );
            }
        } while (empty($agent->getGenomeArray()));

        return $agent;
    }

    /**
     * Run each agent with the data. The count of training rows is the age of the agent
     * @param mixed $fitnessFunction Your custom function to calculate the fitness value for each agent
     * @param array $data [ [[input1, input2, input3], [output1, output2]], [[inputX, inputY, inputZ], [outputA, outputB]] ]
     * @param float $surviveRate How many percent of the population should pass to the next generation
     * @param int $generationCount how many generations to pass
     * @return World
     * @throws Exception
     */
    public function step($fitnessFunction, array $data, int $generationCount = 1, float $surviveRate = 0.9): self
    {
        for ($i = 0; $i < $generationCount; $i++) {
            $this->nextGeneration($fitnessFunction, $data, $surviveRate);
        }

        return $this;
    }

    /**
     * Run each agent with the data. The count of training rows is the age of the agent
     * @param mixed $fitnessFunction Your custom function to calculate the fitness value for each agent
     * @param array $data [ [[input1, input2, input3], [output1, output2]], [[inputX, inputY, inputZ], [outputA, outputB]] ]
     * @param float $surviveRate How many percent of the population should pass to the next generation
     * @return World
     * @throws Exception
     */
    //TODO: add synchronous param: True to run every agent simultaneously with others, False to age 1 agent completely before running the next
    public function nextGeneration($fitnessFunction, array $data, float $surviveRate = 0.9): self
    {
        // Step all agents
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
            $fitnessByAgentKey[] = [
                'fitness' => $fitness,
                'agent_key' => $key
            ];
        }

        // Sort agent keys by their fitness
        usort($fitnessByAgentKey, function($a, $b) {
            return $b['fitness'] > $a['fitness'] ? 1 : -1;
        });

        // Save best agent
        $highestFitness = $fitnessByAgentKey[0]['fitness'];
        // If the best in this generation was better that the best in the previous generations
        if (!$this->bestAgent || $this->bestAgent->getFitness() < $highestFitness) {
            $bestAgentKey = $fitnessByAgentKey[0]['agent_key'];
            $this->bestAgent = $this->agents[$bestAgentKey];
        }

        // Survival
        $survivedCount = round(count($fitnessByAgentKey) * $surviveRate);
        $fitnessByAgentKey = array_slice($fitnessByAgentKey, 0, $survivedCount);

        // Reproduction
        shuffle($fitnessByAgentKey);
        $population = count($this->agents);
        $newAgents = [];
        $i = 0;
        while(count($newAgents) < $population) {
            // Go to first of the array again if out of range
            if ($i > count($fitnessByAgentKey) - 1) {
                $i = 0;
            }
            $agent1 = $this->agents[$fitnessByAgentKey[$i]['agent_key']];
            $i++;
            if ($i > count($fitnessByAgentKey) - 1) {
                $i = 0;
            }
            $agent2 = $this->agents[$fitnessByAgentKey[$i]['agent_key']];
            $i++;

            $newAgents[] = $this->reproduce($agent1, $agent2);
            if (count($newAgents) < $population) {
                $newAgents[] = $this->reproduce($agent1, $agent2);
            }
        }

        $this->setAgents($newAgents);

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

    public function getBestAgent(): ?Agent
    {
        return $this->bestAgent;
    }
}
