<?php

namespace Rotifer\Models;

use Exception;
use Rotifer\GeneEncoders\BinaryEncoder;
use Rotifer\GeneEncoders\Encoder;
use Rotifer\GeneEncoders\HexEncoder;
use Rotifer\GeneEncoders\HumanEncoder;
use Rotifer\Helpers\ReproductionHelper;

class World
{
    private string $name;
    /**
     * @var Agent[]
     */
    private array $agents = [];
    private int $generation = 1;
    private ?Agent $bestAgent = null;

    public function __construct($name = 'default')
    {
        $this->name = $name;

        if (!file_exists('autosave')) {
            mkdir('autosave');
        }
    }

    /**
     * @param int $count How many agents to create
     * @param $inputNeuronsCount
     * @param $outputNeuronsCount
     * @param array $hiddenLayersNeurons For having static agents, for example 3 layers with 4, 5, 4 each: [4, 5, 4]. Leave empty for dynamic size.
     * @param bool $hasMemory Do they have memory or they should be trained for each training row without any previous knowledge
     * @return World
     * @throws Exception
     */
    public function createAgents(int $count, $inputNeuronsCount, $outputNeuronsCount, array $hiddenLayersNeurons = [], bool $hasMemory = false): self
    {
        if ($count <= 1) {
            throw new Exception('The world cannot have only 1 agent.');
        }

        $agents = [];

        if ($hiddenLayersNeurons) {
            // Static agents
            for ($i = 0; $i < $count; $i++) {
                $agent = new StaticAgent();
                $agent->createNeuron(Neuron::TYPE_INPUT, $inputNeuronsCount);
                $agent->createNeuron(Neuron::TYPE_OUTPUT, $outputNeuronsCount);
                $agent->createHiddenLayerNeurons($hiddenLayersNeurons);
                $agent->setHasMemory($hasMemory);
                $agent->initRandomConnections();
                $agents[] = $agent;
            }
        } else {
            // Dynamic agents
            for ($i = 0; $i < $count; $i++) {
                $agent = new Agent();
                $agent->createNeuron(Neuron::TYPE_INPUT, $inputNeuronsCount);
                $agent->createNeuron(Neuron::TYPE_HIDDEN, 1);
                $agent->createNeuron(Neuron::TYPE_OUTPUT, $outputNeuronsCount);
                $agent->setHasMemory($hasMemory);
                $agent->initRandomConnections();
                $agents[] = $agent;
            }
        }

        return $this->setAgents($agents);
    }

    public static function loadAutoSaved($name = 'default', bool $hasMemory = false): self
    {
        if (!file_exists('autosave/world_' . $name . '.txt')) {
            throw new Exception('World file for name: ' . $name . ' does not exist');
        }
        $world = file_get_contents('autosave/world_' . $name . '.txt');
        $world = self::createFromGenomesString($world, HexEncoder::getInstance(), ';', "\n", $hasMemory);
        $world->name = $name;

        // Set best agent
        $world->bestAgent = Agent::loadFromFile($name, HexEncoder::getInstance(), $hasMemory);
        // Add the best agent among other agents
        $world->agents[array_key_last($world->agents)] = $world->bestAgent;

        return $world;
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
     * Run a reproduction tournament
     * @param array $agentAndFitnessArray [[fitness => 221, agentKey => 2], [fitness => 561, agentKey => 3],...]
     * @return array New agents for the new generation. [agent, agent, agent,...]
     * @throws Exception
     */
    public function tournament(array $agentAndFitnessArray): array
    {
        shuffle($agentAndFitnessArray);
        $population = count($this->agents);
        $tournamentCount = ceil($population / 10);
        // Chunk agents to several tournaments
        $tournaments = array_chunk($agentAndFitnessArray, $tournamentCount);
        // Sort tournament agents by their fitness
        foreach ($tournaments as $key => $tournament) {
            usort($tournaments[$key], function($a, $b) {
                return $b['fitness'] > $a['fitness'] ? 1 : -1;
            });
        }
        $tempTournaments = $tournaments;
        $newAgents = [];
        $i = 0;
        while(count($newAgents) < $population) {
            // Run the tournament again if population is not filled
            if ($i == 0 && count($tempTournaments[$i]) == 0) {
                $tempTournaments = $tournaments;
            }

            // If last tournament doesn't contain any agent
            if ($i == array_key_last($tempTournaments) && count($tempTournaments[$i]) == 0) {
                continue;
            }

            // If tournament only contains 1 agent
            if (count($tempTournaments[$i]) == 1) {
                $agentKey = $tempTournaments[$i][array_key_first($tempTournaments[$i])]['agent_key'];
                $newAgents[] = $this->agents[$agentKey];
                unset($tempTournaments[$i][array_key_first($tempTournaments[$i])]);
                continue;
            }

            // Get 2 first ones from the tournament
            $firstTournamentAgentKey = array_key_first($tempTournaments[$i]);
            $agent1 = $this->agents[$tempTournaments[$i][$firstTournamentAgentKey]['agent_key']];
            unset($tempTournaments[$i][$firstTournamentAgentKey]);

            $secondTournamentAgentKey = array_key_first($tempTournaments[$i]);
            $agent2 = $this->agents[$tempTournaments[$i][$secondTournamentAgentKey]['agent_key']];
            unset($tempTournaments[$i][$secondTournamentAgentKey]);

            // Reproduce 2 children
            $newAgents[] = $this->reproduce($agent1, $agent2);
            if (count($newAgents) < $population) {
                $newAgents[] = $this->reproduce($agent1, $agent2);
            }

            $i++;
            if ($i > array_key_last($tempTournaments)) {
                $i = 0; //rewind
            }
        }

        return $newAgents;
    }

    /**
     * Reproduce children from 2 agents
     * @param Agent $agent1
     * @param Agent $agent2
     * @return Agent
     * @throws Exception
     */
    public function reproduce(Agent $agent1, Agent $agent2): Agent
    {
        // Do the reproduction process and if there is no gene, repeat the reproduction
        $attempts = 0;
        do {
            // Crossover
            $newAgent = ReproductionHelper::crossover($agent1, $agent2, PROBABILITY_CROSSOVER);

            // Mutation
            $newAgent = ReproductionHelper::mutate($newAgent, PROBABILITY_MUTATE_WEIGHT, PROBABILITY_MUTATE_ADD_NEURON, PROBABILITY_MUTATE_ADD_GENE, PROBABILITY_MUTATE_REMOVE_NEURON, PROBABILITY_MUTATE_REMOVE_GENE);

            // Make a fresh agent (unique connections, remove all neuron values and reset step)
            $hasMemory = $newAgent->hasMemory();
            if ($agent1 instanceof StaticAgent) {
                /** @var StaticAgent $newAgent */
                $newAgent = StaticAgent::createFromGenome($newAgent->getGenomeArray(), $hasMemory);
                $newAgent->setLayers($agent1->getLayers());
            } else {
                $newAgent = Agent::createFromGenome($newAgent->getGenomeArray(), $hasMemory);
            }

            $attempts++;
            if ($attempts > 100) {
                //return $agent1;
                throw new Exception('Tried ' . $attempts
                    . ' times to reproduce but it always generated agent with no genes. genome1: '
                    . $agent1->getGenomeString(HumanEncoder::getInstance())
                    . ' genome2: ' . $agent2->getGenomeString(HumanEncoder::getInstance())
                );
            }
        } while (empty($newAgent->getGenomeArray()));

        return $newAgent;
    }

    /**
     * Run each agent with the data. The count of training rows is the age of the agent
     * @param mixed $fitnessFunction Your custom function to calculate the fitness value for each agent
     * @param array $data [ [[input1, input2, input3], [output1, output2]], [[inputX, inputY, inputZ], [outputA, outputB]] ]
     * @param int $generationCount how many generations to pass, 0 for unlimited
     * @param float $surviveRate How many percent of the population should pass to the next generation
     * @param mixed $stopFunction The minimum fitness that eliminates the world
     * @param int $batchSize 0 means the batch size is equal to the whole data size
     * @return World
     * @throws Exception
     */
    public function step($fitnessFunction, array $data, int $generationCount = 1, float $surviveRate = 0.9, int $batchSize = 0, $stopFunction = null): self
    {
        $dataCount = count($data);
        $dataChunkCount = ($batchSize == 0 || $batchSize >= $dataCount) ? $dataCount : ceil($dataCount / $batchSize);

        while ($generationCount == 0 || $this->generation <= $generationCount) {
            if ($batchSize > 0 && $batchSize != $dataCount) {
                $batchStartIndex = ($this->generation - 1) % $dataChunkCount  * $batchSize;
                $this->nextGeneration($fitnessFunction, array_slice($data, $batchStartIndex, $batchSize), $surviveRate);
            } else {
                $this->nextGeneration($fitnessFunction, $data, $surviveRate);
            }
            $this->generation++;

            // If best fitness passes stop fitness, stop the world
            if (!is_null($stopFunction) && $stopFunction($this)) {
                return $this;
            }
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
            // Reset any previous memory and fitness
            $this->agents[$key]->reset();

            // Each data
            $fitness = 0;
            foreach ($data as $row) {
                // Empty data means memory reset
                if (empty($row)) {
                    $this->agents[$key]->resetMemory();
                    continue;
                }

                // Step
                $this->agents[$key]->step($row[0]);

                // Feed data into fitness function
                $otherAgents = $this->getAgents();
                unset($otherAgents[$key]);
                $fitness += $fitnessFunction($this->agents[$key], $row, $otherAgents, $this);
            }
            $this->agents[$key]->setFitness($fitness);
            $fitnessByAgentKey[] = [
                'fitness' => $fitness,
                'agent_key' => $key
            ];
            if (!in_array('--quiet', $_SERVER['argv'] ?? [])) {
                if (!isset($percent)) $percent = 1;
                echo 'Generation ' . $this->generation . ' - ' . min(100, str_pad(round(($percent++) * 100 / count($this->agents), 1), 4)) . "%\r";
                flush();
            }
        }

        // Sort agent keys by their fitness
        usort($fitnessByAgentKey, function($a, $b) {
            return $b['fitness'] > $a['fitness'] ? 1 : -1;
        });

        // Save best agent
        $improved = false;
        $highestFitness = $fitnessByAgentKey[0]['fitness'];
        // If the best in this generation was better that the best in the previous generations
        if (empty($this->bestAgent) || $this->bestAgent->getFitness() < $highestFitness) {
            $bestAgentKey = $fitnessByAgentKey[0]['agent_key'];
            // Save best agent in world instance
            $this->bestAgent = $this->agents[$bestAgentKey];
            $improved = true;

            // Save best agent in file
            file_put_contents('autosave/best_agent_' . $this->name . '.txt', $this->bestAgent->getGenomeString(HexEncoder::getInstance()));
        }

        // Save world in file
        if (SAVE_WORLD_EVERY_GENERATION !== 0 && $this->generation % SAVE_WORLD_EVERY_GENERATION == 0) {
            file_put_contents('autosave/world_' . $this->name . '.txt', $this->getGenomesString(HexEncoder::getInstance()));
        }

        if (!in_array('--quiet', $_SERVER['argv'] ?? [])) {
            echo 'Generation ' . $this->generation . ' - Best in generation: ' . str_pad($highestFitness, 15)
                . ' - Best overall: ' . str_pad($this->bestAgent?->getFitness() ?? 0, 15)
                . ' - Genes: ' . count($this->bestAgent->getGenomeArray())
                . ' - H.Neurons: ' . count($this->bestAgent->getNeuronsByType(Neuron::TYPE_HIDDEN))
                . ($this->generation != 1 && $improved ? ' - Improved' : null) . PHP_EOL;
            flush();
        }

        // Survival
        $survivedCount = round(count($fitnessByAgentKey) * $surviveRate);
        $fitnessByAgentKey = array_slice($fitnessByAgentKey, 0, $survivedCount);

        // Reproduction
        $newAgents = $this->tournament($fitnessByAgentKey);

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

    public static function createFromGenomesString($genomes, Encoder $decoder = null, $geneSeparator = ';', $genomeSeparator = "\n", bool $hasMemory = false): self
    {
        $world = new self();

        if (is_null($decoder)) {
            $decoder = BinaryEncoder::getInstance();
        }

        $genomes = explode($genomeSeparator, $genomes);

        $agents = [];
        foreach ($genomes as $genome) {
            $agents[] = Agent::createFromGenome($genome, $hasMemory, $decoder, $geneSeparator);
        }

        $world->setAgents($agents);

        return $world;
    }

    public function getBestAgent(): ?Agent
    {
        return $this->bestAgent;
    }
}
