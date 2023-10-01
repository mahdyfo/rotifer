<?php

namespace Rotifer\Helpers;

use Rotifer\Models\Agent;
use Rotifer\Models\Neuron;
use Rotifer\Models\StaticAgent;

class ReproductionHelper
{
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
    public static function mutate(Agent $agent, float $changeGeneProbability = 0.5, float $addNeuronProbability = 0.3, float $addConnectionProbability = 0.3, float $deleteNeuronProbability = 0.1, float $deleteConnectionProbability = 0.1): Agent
    {
        // 1. Add neuron
        if (mt_rand(1, 10000) / 10000 <= $addNeuronProbability) {
            $newNeuron = $agent->createNeuron(Neuron::TYPE_HIDDEN);

            // Connect input to new neuron
            $randomInput = $agent->getRandomNeuronByType(Neuron::TYPE_INPUT);
            $agent->connectNeurons($randomInput, $newNeuron, WeightHelper::generateRandomWeight());

            // Connect the new neuron to a random output
            $randomOutput = $agent->getRandomNeuronByType(Neuron::TYPE_OUTPUT);
            $agent->connectNeurons($newNeuron, $randomOutput, WeightHelper::generateRandomWeight());

            $agent->deleteRedundantGenes();
        }

        // 2. Add connection
        if (mt_rand(1, 10000) / 10000 <= $addConnectionProbability) {
            $genome = $agent->getGenomeArray();

            // Current connections
            $currentConnections = [];
            foreach ($genome as $gene) {
                $currentConnections[] = $gene['from_type'] . '.' . $gene['from_index'] . '.' . $gene['to_type'] . '.' . $gene['to_index'];
            }

            // All possible connections
            $possibleConnections = [];
            $inputs = $agent->getNeuronsByType(Neuron::TYPE_INPUT);
            $hiddens = $agent->getNeuronsByType(Neuron::TYPE_HIDDEN);
            $outputs = $agent->getNeuronsByType(Neuron::TYPE_OUTPUT);
            // Input -> Hidden / Input -> Output
            foreach ($inputs as $index => $input) {
                foreach ($hiddens as $hidden) {
                    $possibleConnections[] = Neuron::TYPE_INPUT . '.' . $index . '.' . Neuron::TYPE_HIDDEN . '.' . $hidden->getIndex();
                }
                foreach ($outputs as $output) {
                    $possibleConnections[] = Neuron::TYPE_INPUT . '.' . $index . '.' . Neuron::TYPE_OUTPUT . '.' . $output->getIndex();
                }
            }
            // Hidden -> Hidden / Hidden -> Output
            foreach ($hiddens as $index => $hidden) {
                foreach ($hiddens as $hidden2) {
                    // Don't include future feedbacks
                    if ($hidden->getIndex() <= $hidden2->getIndex()) {
                        $possibleConnections[] = Neuron::TYPE_HIDDEN . '.' . $index . '.' . Neuron::TYPE_HIDDEN . '.' . $hidden2->getIndex();
                    }
                }
                foreach ($outputs as $output) {
                    $possibleConnections[] = Neuron::TYPE_HIDDEN . '.' . $index . '.' . Neuron::TYPE_OUTPUT . '.' . $output->getIndex();
                }
            }

            // Create a new connection
            $diffConnections = array_diff($possibleConnections, $currentConnections);
            if ($diffConnections) {
                shuffle($diffConnections);
                $diffConnection = explode('.', $diffConnections[array_rand($diffConnections)]);
                $agent->connectNeurons(
                    $agent->findNeuron($diffConnection[0], $diffConnection[1]),
                    $agent->findNeuron($diffConnection[2], $diffConnection[3]),
                    WeightHelper::generateRandomWeight()
                );
            }
        }

        // 3. Change weight
        if (mt_rand(1, 10000) / 10000 <= $changeGeneProbability) {
            $genome = $agent->getGenomeArray();
            for ($i = 0; $i < MUTATE_WEIGHT_COUNT ?? 1; $i++) {
                $genome[array_rand($genome)]['weight'] = WeightHelper::generateRandomWeight();
            }
            $agent->setGenome($genome);
        }

        // 4. Delete neuron
        if (mt_rand(1, 10000) / 10000 <= $deleteNeuronProbability) {
            $hiddens = $agent->getNeuronsByType(Neuron::TYPE_HIDDEN);

            // There should be at least one hidden neuron to delete
            if (!empty($hiddens)) {
                $excludeHiddenIndexes = [];
                $inputs = $agent->getNeuronsByType(Neuron::TYPE_INPUT);
                $outputs = $agent->getNeuronsByType(Neuron::TYPE_OUTPUT);

                // Find neurons with 1 connection from inputs
                foreach ($inputs as $input) {
                    $connections = $input->getOutConnections();
                    if (!empty($connections[Neuron::TYPE_HIDDEN]) && count($connections[Neuron::TYPE_HIDDEN]) == 1) {
                        $excludeHiddenIndexes[] = array_keys($connections[Neuron::TYPE_HIDDEN])[0];
                    }
                }
                // Find neurons with 1 connection to outputs
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

            $agent->deleteRedundantGenes();
        }

        // 5. Delete connection
        if (mt_rand(1, 10000) / 10000 <= $deleteConnectionProbability) {
            $genome = $agent->getGenomeArray();
            unset($genome[array_rand($genome)]);
            $agent->setGenome($genome);

            // No inputs or outputs should be without connection
            $inputs = $agent->getNeuronsByType(Neuron::TYPE_INPUT);
            $outputs = $agent->getNeuronsByType(Neuron::TYPE_OUTPUT);
            foreach($inputs as $input) {
               if (count($input->getOutConnections()) == 0) {
                   $agent->connectToAll($input);
               }
            }
            foreach($outputs as $output) {
               if (count($output->getInConnections()) == 0) {
                   $agent->connectToAll($output);
               }
            }

            $agent->deleteRedundantGenes();
        }

        return $agent;
    }
}
