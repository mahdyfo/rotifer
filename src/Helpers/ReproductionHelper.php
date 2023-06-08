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
     * @param array $genome1
     * @param array $genome2
     * @param float $probability
     * @return array Two new genomes
     */
    public static function crossover(array $genome1, array $genome2, float $probability = 0.2): array
    {
        // AAA BBB CCC DDD EEE
        // FFF GGG HHH III JJJ KKK
        // Results:
        // AFA BGG HCC IDI JEJ
        // FAF GBB CHH DID EJE KKK
        foreach ($genome1 as $geneIndex => $gene) {
            // If the other genome is shorter, continue
            if (!isset($genome2[$geneIndex])) {
                continue;
            }

            // Continue if the probability is not met
            if (mt_rand(1, 100) / 100 > $probability) {
                continue;
            }

            $crossoverPlaces = [/*'from', 'to',*/ 'weight'];
            foreach ($crossoverPlaces as $crossoverPlace) {
                // Swap
                switch ($crossoverPlace) {
                    // Swap from_type and from_index
                    case 'from':
                        $tempSwapVar = $genome2[$geneIndex]['from_type'];
                        $genome2[$geneIndex]['from_type'] = $genome1[$geneIndex]['from_type'];
                        $genome1[$geneIndex]['from_type'] = $tempSwapVar;
                        $tempSwapVar = $genome2[$geneIndex]['from_index'];
                        $genome2[$geneIndex]['from_index'] = $genome1[$geneIndex]['from_index'];
                        $genome1[$geneIndex]['from_index'] = $tempSwapVar;
                        break;
                    // Swap to_type and to_index
                    case 'to':
                        $tempSwapVar = $genome2[$geneIndex]['to_type'];
                        $genome2[$geneIndex]['to_type'] = $genome1[$geneIndex]['to_type'];
                        $genome1[$geneIndex]['to_type'] = $tempSwapVar;
                        $tempSwapVar = $genome2[$geneIndex]['to_index'];
                        $genome2[$geneIndex]['to_index'] = $genome1[$geneIndex]['to_index'];
                        $genome1[$geneIndex]['to_index'] = $tempSwapVar;
                        break;
                    // Swap weights
                    case 'weight':
                        $tempSwapVar = $genome2[$geneIndex]['weight'];
                        $genome2[$geneIndex]['weight'] = $genome1[$geneIndex]['weight'];
                        $genome1[$geneIndex]['weight'] = $tempSwapVar;
                        break;
                    default:
                        break;
                }
            }
        }

        return [$genome1, $genome2];
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