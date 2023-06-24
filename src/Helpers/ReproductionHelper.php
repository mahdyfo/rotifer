<?php

namespace GeneticAutoml\Helpers;

use GeneticAutoml\Models\Agent;
use GeneticAutoml\Models\Neuron;

class ReproductionHelper
{
    /**
     * @param array $genome1
     * @param array $genome2
     * @param float $lowestDominance
     * @param float $highestDominance
     * @return array New genome
     */
    public static function dominance(array $genome1, array $genome2, float $lowestDominance = 0.3, float $highestDominance = 0.7): array
    {
        $genomes = [$genome1, $genome2];
        shuffle($genomes);

        $crossPoint = mt_rand($lowestDominance * 100, $highestDominance * 100) / 100;
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

        return array_merge($part1, $part2);
    }

    /**
     * Makes two new genomes that are crossed over.
     * Crossover is possible in from-connection / to-connection / weight
     * @param Agent $agent1
     * @param Agent $agent2
     * @param float $probability
     * @return Agent Two new genomes
     * @throws \Exception
     */
    public static function crossover(Agent $agent1, Agent $agent2, float $probability = 0.2): Agent
    {
        // Agent1: AA[A]  BBB  CC[C]  DDD  EE[E]
        // Agent2: FF[F]  GGG  HH[H]  III  JJ[J]
        // Result:
        // Agentx: AA F   BBB  CC H   DDD  EE J (Changed agent1)

        $newAgent = Agent::createFromGenome($agent1->getGenomeArray());

        $genome2 = $agent2->getGenomeArray();
        // Iterate through all connections of agent2
        foreach ($genome2 as $gene) {
            // Continue if the probability is not met
            if (mt_rand(1, 100) / 100 > $probability) {
                continue;
            }

            // Try to find the connections of agent2 in agent1
            // Find agent2 neuron origin in agent1
            $originNeuronInAgent1 = $newAgent->findNeuron($gene['from_type'], $gene['from_index']);
            // If the same neuron origin exist in the agent1
            if (!empty($originNeuronInAgent1)) {
                // Find out if neuron destination exist in agent1 too
                $destNeuronInAgent1 = $newAgent->findNeuron($gene['to_type'], $gene['to_index']);
                if (!empty($destNeuronInAgent1)) {
                    // Set weight of agent2 connection in agent1 connection or create the connection if there is none
                    $newAgent = $newAgent->connectNeurons($originNeuronInAgent1, $destNeuronInAgent1, $gene['weight']);
                }
            }
        }

        $newAgent->deleteRedundantGenes();

        return $newAgent;
    }

    /**
     *  Mutation types: 1. Add neuron, 2. Delete neuron, 3. Change weights
     * @param Agent $agent
     * @param float $changeGeneProbability
     * @param float $addNeuronProbability
     * @param float $deleteNeuronProbability
     * @return Agent
     * @throws \Exception
     */
    public static function mutate(Agent $agent, float $changeGeneProbability = 0.5, float $addNeuronProbability = 0.3, float $deleteNeuronProbability = 0.1): Agent
    {
        // 1. Add neuron
        if (mt_rand(1, 100) / 100 <= $addNeuronProbability) {
            $agent->createNeuron(Neuron::TYPE_HIDDEN, 1, true);
        }

        // 2. Change weight
        $genome = $agent->getGenomeArray();
        $changedGenome = false;
        foreach ($genome as $key => $gene) {
            // Continue if the probability is not met
            if (mt_rand(1, 100) / 100 > $changeGeneProbability) {
                continue;
            }

            $genome[$key]['weight'] = WeightHelper::generateRandomWeight();
            $changedGenome = true;
        }
        if ($changedGenome) {
            $agent->setGenome($genome);
        }

        // 3. Delete neuron
        if (mt_rand(1, 100) / 100 <= $deleteNeuronProbability) {
            $hiddens = $agent->getNeuronsByType(Neuron::TYPE_HIDDEN);

            // There should be at least one hidden neuron to delete
            if (!empty($hiddens)) {
                $excludeHiddenIndexes = [];
                $inputs = $agent->getNeuronsByType(Neuron::TYPE_INPUT);
                $outputs = $agent->getNeuronsByType(Neuron::TYPE_OUTPUT);

                // Find neurons of 1 connection inputs
                foreach ($inputs as $input) {
                    $connections = $input->getOutConnections();
                    if (!empty($connections[Neuron::TYPE_HIDDEN]) && count($connections[Neuron::TYPE_HIDDEN]) == 1) {
                        $excludeHiddenIndexes[] = array_keys($connections[Neuron::TYPE_HIDDEN])[0];
                    }
                }
                // Find neurons of 1 connection outputs
                foreach ($outputs as $output) {
                    $connections = $output->getInConnections();
                    if (!empty($connections[Neuron::TYPE_HIDDEN]) && count($connections[Neuron::TYPE_HIDDEN]) == 1) {
                        $excludeHiddenIndexes[] = array_keys($connections[Neuron::TYPE_HIDDEN])[0];
                    }
                }
                // Exclude neurons that their removal causes input or output to be without connections
                foreach ($excludeHiddenIndexes as $excludeHiddenIndex) {
                    unset($hiddens[$excludeHiddenIndex]);
                }
                if (!empty($hiddens)) {
                    // Remove a random neuron that doesn't leave inputs or outputs without connections
                    $hiddenIndexes = array_keys($hiddens);
                    $randomHiddenIndex = $hiddenIndexes[array_rand($hiddenIndexes)];
                    $agent->removeNeuron(Neuron::TYPE_HIDDEN, $randomHiddenIndex);
                }
            }
        }

        $agent->deleteRedundantGenes();

        return $agent;
    }

    /**
     * Move gene: Translocation or change position of a gene in the genome
     * @param array $genome
     * @param float $probability
     * @return array Genome
     */
    public static function translocation(array $genome, float $probability = 0.2): array
    {
        if (count($genome) <= 1 || mt_rand(1, 100) / 100 > $probability) {
            return $genome;
        }

        $randomIndexes = array_rand($genome, 2);
        $tempSwapVar = $genome[$randomIndexes[0]];
        $genome[$randomIndexes[0]] = $genome[$randomIndexes[1]];
        $genome[$randomIndexes[1]] = $tempSwapVar;

        return $genome;
    }
}