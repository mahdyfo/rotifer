<?php

namespace GeneticAutoml\Helpers;

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

            $crossoverPlaces = ['from', 'to', 'weight'];
            foreach ($crossoverPlaces as $crossoverPlace) {
                // Continue if the probability is not met
                if (mt_rand(1, 100) / 100 > $probability) {
                    continue;
                }

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

    public static function mutate(array $genome, float $changeGeneProbability = 0.2, float $addGeneProbability = 0.2, float $deleteGeneProbability = 0.1): array
    {
        // TODO: change genome to agent. So we can have more control over connections and neurons. The mutation will be more gene-aware.
        // Mutation types:
        // 1. Delete gene
        // 2. Change gene:
        //    change from_index
        //    change to_index
        //    change weight
        // 3. Add gene:
        //    1 connection : Input to Neuron, Neuron to Neuron, Neuron to Output
        //    2 connections: Add a neuron (Input to New-Neuron to Output, Input to New-Neuron to Neuron, Neuron to New-Neuron to Output, Neuron to New-Neuron to Neuron)

        // Gather all indexes
        $indexes = [];
        foreach ($genome as $gene) {
            $indexes[$gene['from_type']][$gene['from_index']] = $gene['from_index'];
            $indexes[$gene['to_type']][$gene['to_index']] = $gene['to_index'];
        }

        // 1. Delete gene
        if (count($genome) > 1 && mt_rand(1, 100) / 100 <= $deleteGeneProbability) {
            unset($genome[array_rand($genome)]);
        }

        // 2. Change gene
        foreach ($genome as $index => $gene) {
            // Continue if the probability is not met
            if (mt_rand(1, 100) / 100 > $changeGeneProbability) {
                continue;
            }

            // Get one change randomly
            $change = ['weight']; // In any case, weight can be mutated
            if (count($indexes[$gene['from_type']] ?? []) > 1) {
                // If there is 1 from-neuron, skip mutation for from. because there is nothing to change to
                $change[] = 'from';
            }
            if (count($indexes[$gene['to_type']] ?? []) > 1) {
                // If there is 1 to-neuron, skip mutation for to. because there is nothing to change to
                $change[] = 'to';
            }
            $change = $change[array_rand($change)];

            switch ($change) {
                case 'from':
                    // Copy indexes in a temp var
                    $fromIndexes = $indexes[$gene['from_type']];
                    // Delete the current gene index from the pool (to avoid selecting the same index again)
                    if (($key = array_search($gene['from_index'], $fromIndexes)) !== false) {
                        unset($fromIndexes[$key]);
                    }
                    $genome[$index]['from_index'] = $fromIndexes[array_rand($fromIndexes)];
                    break;
                case 'to':
                    // Copy indexes in a temp var
                    $toIndexes = $indexes[$gene['to_type']];
                    // Delete the current gene index from the pool (to avoid selecting the same index again)
                    if (($key = array_search($gene['to_index'], $toIndexes)) !== false) {
                        unset($toIndexes[$key]);
                    }
                    $genome[$index]['to_index'] = $toIndexes[array_rand($toIndexes)];
                    break;
                case 'weight':
                    $genome[$index]['weight'] = WeightHelper::generateRandomWeight();
                    break;
                default:
                    break;
            }
        }

        // 3. Add gene
        if (mt_rand(1, 100) / 100 <= $addGeneProbability) {
            // Get one change randomly
            $change = ['1-connection', '2-connection'];
            $change = $change[array_rand($change)];

            switch ($change) {
                case '1-connection':
                    if (!empty($indexes[Neuron::TYPE_INPUT])) {
                        // Has any input neurons
                        $newGene['from_type'][] = Neuron::TYPE_INPUT;
                    }
                    if (!empty($indexes[Neuron::TYPE_HIDDEN])) {
                        $newGene['from_type'][] = Neuron::TYPE_HIDDEN;
                        $newGene['to_type'][] = Neuron::TYPE_HIDDEN;
                    }
                    if (!empty($indexes[Neuron::TYPE_OUTPUT])) {
                        $newGene['to_type'][] = Neuron::TYPE_OUTPUT;
                    }
                    $newGene['from_type'] = $newGene['from_type'][array_rand($newGene['from_type'])];
                    $newGene['to_type'] = $newGene['to_type'][array_rand($newGene['to_type'])];

                    $newGene['from_index'] = $indexes[$newGene['from_type']][array_rand($indexes[$newGene['from_type']])];
                    $newGene['to_index'] = $indexes[$newGene['to_type']][array_rand($indexes[$newGene['to_type']])];
                    $newGene['weight'] = WeightHelper::generateRandomWeight();
                    $genome[] = $newGene;
                    break;
                case '2-connection':
                    // New neuron

                    // Connection 1:
                    // From input to new hidden neuron
                    $newGene1['from_type'] = Neuron::TYPE_INPUT;
                    $newGene1['from_index'] = !empty($indexes[Neuron::TYPE_INPUT])
                        ? $indexes[Neuron::TYPE_INPUT][array_rand($indexes[Neuron::TYPE_INPUT])]
                        : 0; // Any network has at least 1 input
                    // To a new hidden
                    $newGene1['to_type'] = Neuron::TYPE_HIDDEN;
                    $newGene1['to_index'] = !empty($indexes[Neuron::TYPE_HIDDEN]) ? array_key_last($indexes[Neuron::TYPE_HIDDEN]) : 0;
                    $newGene1['weight'] = WeightHelper::generateRandomWeight();

                    // Connection 2:
                    // From the newly created hidden to output
                    $newGene2['from_type'] = $newGene1['to_type'];
                    $newGene2['from_index'] = $newGene1['to_index'];
                    // To output
                    $newGene2['to_type'] = Neuron::TYPE_OUTPUT;
                    $newGene2['to_index'] = !empty($indexes[Neuron::TYPE_OUTPUT])
                        ? $indexes[Neuron::TYPE_OUTPUT][array_rand($indexes[Neuron::TYPE_OUTPUT])]
                        : 0; // Any network has at least 1 output
                    $newGene2['weight'] = WeightHelper::generateRandomWeight();

                    $genome[] = $newGene1;
                    $genome[] = $newGene2;
                    break;
            }
        }

        return $genome;
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