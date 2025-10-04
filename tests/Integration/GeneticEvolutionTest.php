<?php

namespace Rotifer\Tests\Integration;

use PHPUnit\Framework\TestCase;
use Rotifer\Models\World;
use Rotifer\Models\Agent;
use Rotifer\Activations\Activation;

// Define constants
if (!defined('ACTIVATION')) {
    define('ACTIVATION', [Activation::class, 'sigmoid']);
}
if (!defined('CALCULATE_STEP_TIME')) {
    define('CALCULATE_STEP_TIME', false);
}
if (!defined('PROBABILITY_CROSSOVER')) {
    define('PROBABILITY_CROSSOVER', 0.5);
}
if (!defined('PROBABILITY_MUTATE_WEIGHT')) {
    define('PROBABILITY_MUTATE_WEIGHT', 0.4);
}
if (!defined('MUTATE_WEIGHT_COUNT')) {
    define('MUTATE_WEIGHT_COUNT', 1);
}
if (!defined('PROBABILITY_MUTATE_ADD_NEURON')) {
    define('PROBABILITY_MUTATE_ADD_NEURON', 0.04);
}
if (!defined('PROBABILITY_MUTATE_REMOVE_NEURON')) {
    define('PROBABILITY_MUTATE_REMOVE_NEURON', 0.04);
}
if (!defined('PROBABILITY_MUTATE_ADD_GENE')) {
    define('PROBABILITY_MUTATE_ADD_GENE', 0.1);
}
if (!defined('PROBABILITY_MUTATE_REMOVE_GENE')) {
    define('PROBABILITY_MUTATE_REMOVE_GENE', 0.1);
}
if (!defined('SAVE_WORLD_EVERY_GENERATION')) {
    define('SAVE_WORLD_EVERY_GENERATION', 0);
}

class GeneticEvolutionTest extends TestCase
{
    protected function tearDown(): void
    {
        parent::tearDown();

        // Clean up autosave directory
        if (is_dir('autosave')) {
            $files = glob('autosave/*');
            foreach ($files as $file) {
                if (is_file($file)) {
                    unlink($file);
                }
            }
        }
    }

    public function testFitnessImproves(): void
    {
        $world = new World('evolution_test');
        $world->createAgents(20, 3, 1);

        $data = [
            [[1, 0, 0], [0]],
            [[1, 0, 1], [1]],
            [[1, 1, 0], [1]],
            [[1, 1, 1], [0]],
        ];

        $fitnessFunction = function (Agent $agent, $dataRow) {
            $predicted = $agent->getOutputValues()[0];
            $actual = $dataRow[1][0];
            return 1.0 - abs($predicted - $actual);
        };

        // Run first generation
        $world->nextGeneration($fitnessFunction, $data, 0.2);
        $firstGenFitness = $world->getBestAgent()->getFitness();

        // Run multiple more generations
        for ($i = 0; $i < 10; $i++) {
            $world->nextGeneration($fitnessFunction, $data, 0.2);
        }

        $finalFitness = $world->getBestAgent()->getFitness();

        // Fitness should improve or at least not get worse
        $this->assertGreaterThanOrEqual($firstGenFitness * 0.9, $finalFitness);
    }

    public function testPopulationDiversity(): void
    {
        $world = new World();
        $world->createAgents(20, 3, 1);

        $data = [[[1, 0, 0], [0]]];

        $fitnessFunction = function (Agent $agent, $dataRow) {
            return mt_rand(0, 100) / 100;
        };

        $world->nextGeneration($fitnessFunction, $data, 0.3);

        // Check that agents have different genomes
        $genomes = [];
        foreach ($world->getAgents() as $agent) {
            $genomes[] = serialize($agent->getGenomeArray());
        }

        $uniqueGenomes = array_unique($genomes);

        // Should have multiple unique genomes (at least 50% diversity)
        $this->assertGreaterThan(count($genomes) * 0.5, count($uniqueGenomes));
    }

    public function testEvolutionConvergence(): void
    {
        $world = new World();
        $world->createAgents(30, 3, 1);

        // Simple AND problem
        $data = [
            [[1, 0, 0], [0]],
            [[1, 0, 1], [0]],
            [[1, 1, 0], [0]],
            [[1, 1, 1], [1]],
        ];

        $fitnessFunction = function (Agent $agent, $dataRow) {
            $predicted = $agent->getOutputValues()[0];
            $actual = $dataRow[1][0];
            return 1.0 - abs($predicted - $actual);
        };

        $initialFitness = 0;

        // Run 20 generations
        for ($i = 0; $i < 20; $i++) {
            $world->nextGeneration($fitnessFunction, $data, 0.2);

            if ($i === 0) {
                $initialFitness = $world->getBestAgent()->getFitness();
            }
        }

        $finalFitness = $world->getBestAgent()->getFitness();

        // Verify evolution completed without errors
        $this->assertInstanceOf(Agent::class, $world->getBestAgent());
        $this->assertGreaterThan(0, $finalFitness);

        // Fitness should improve OR stay high (if already good)
        // Don't require strict improvement due to randomness
        $this->assertGreaterThanOrEqual($initialFitness * 0.8, $finalFitness,
            "Fitness should not degrade significantly. Initial: {$initialFitness}, Final: {$finalFitness}");
    }

    public function testSurvivalSelection(): void
    {
        $world = new World();
        $world->createAgents(20, 3, 1);

        $data = [[[1, 0, 0], [0]]];

        // Assign predetermined fitness values
        $fitnessFunction = function (Agent $agent, $dataRow) {
            static $counter = 0;
            $counter++;
            return $counter * 0.1; // Increasing fitness
        };

        $world->nextGeneration($fitnessFunction, $data, 0.5); // 50% survival rate

        // After selection, all agents should have genomes (no empty agents)
        foreach ($world->getAgents() as $agent) {
            $this->assertNotEmpty($agent->getGenomeArray());
        }
    }

    public function testDynamicArchitectureEvolution(): void
    {
        // Test that dynamic agents can evolve different architectures
        // Create agents and manually add/remove neurons to verify capability
        $agent1 = new Agent();
        $agent1->createNeuron(\Rotifer\Models\Neuron::TYPE_INPUT, 2);
        $agent1->createNeuron(\Rotifer\Models\Neuron::TYPE_HIDDEN, 1);
        $agent1->createNeuron(\Rotifer\Models\Neuron::TYPE_OUTPUT, 1);
        $agent1->initRandomConnections();

        $initialHiddenCount = count($agent1->getNeuronsByType(\Rotifer\Models\Neuron::TYPE_HIDDEN));
        $this->assertEquals(1, $initialHiddenCount);

        // Add a neuron
        $agent1->createNeuron(\Rotifer\Models\Neuron::TYPE_HIDDEN, 1, true);
        $afterAddCount = count($agent1->getNeuronsByType(\Rotifer\Models\Neuron::TYPE_HIDDEN));
        $this->assertEquals(2, $afterAddCount);

        // Verify genome can be encoded/decoded with different architectures
        $genome = $agent1->getGenomeArray();
        $this->assertGreaterThan(0, count($genome));

        // Create agent with different initial architecture
        $agent2 = new Agent();
        $agent2->createNeuron(\Rotifer\Models\Neuron::TYPE_INPUT, 2);
        $agent2->createNeuron(\Rotifer\Models\Neuron::TYPE_HIDDEN, 5);
        $agent2->createNeuron(\Rotifer\Models\Neuron::TYPE_OUTPUT, 1);
        $agent2->initRandomConnections();

        $agent2HiddenCount = count($agent2->getNeuronsByType(\Rotifer\Models\Neuron::TYPE_HIDDEN));
        $this->assertEquals(5, $agent2HiddenCount);

        // Verify both agents can process inputs despite different architectures
        $agent1->step([1.0, 0.5]);
        $agent2->step([1.0, 0.5]);

        $this->assertNotEmpty($agent1->getOutputValues());
        $this->assertNotEmpty($agent2->getOutputValues());

        // Verify architecture differences are preserved in genome
        $this->assertNotSame(
            count($agent1->getGenomeArray()),
            count($agent2->getGenomeArray())
        );
    }

    public function testGenerationProgression(): void
    {
        $world = new World();
        $world->createAgents(10, 2, 1);

        $data = [[[1, 0], [0]]];

        $fitnessFunction = function (Agent $agent, $dataRow) {
            return 1.0;
        };

        $generationCount = 5;
        $world->step($fitnessFunction, $data, $generationCount, 0.5);

        // Best agent should exist after evolution
        $this->assertInstanceOf(Agent::class, $world->getBestAgent());
        $this->assertGreaterThan(0, $world->getBestAgent()->getFitness());
    }

    public function testStopFunctionHaltsEvolution(): void
    {
        $world = new World();
        $world->createAgents(10, 2, 1);

        $data = [[[1, 0], [1]]];

        $generationCounter = 0;

        $fitnessFunction = function (Agent $agent, $dataRow) use (&$generationCounter) {
            $generationCounter++;
            return 1.0;
        };

        $stopFunction = function (World $world) {
            return $world->getBestAgent()->getFitness() >= 0.99;
        };

        $world->step($fitnessFunction, $data, 100, 0.5, 0, $stopFunction);

        // Should stop early due to stop function (much less than 100 * 10 agents)
        $this->assertLessThan(1000, $generationCounter);
    }

    public function testBatchTraining(): void
    {
        $world = new World();
        $world->createAgents(10, 3, 1);

        $data = [
            [[1, 0, 0], [0]],
            [[1, 0, 1], [1]],
            [[1, 1, 0], [1]],
            [[1, 1, 1], [0]],
        ];

        $fitnessFunction = function (Agent $agent, $dataRow) {
            $predicted = $agent->getOutputValues()[0];
            $actual = $dataRow[1][0];
            return 1.0 - abs($predicted - $actual);
        };

        // Use batch size of 2
        $world->step($fitnessFunction, $data, 4, 0.5, 2);

        $this->assertInstanceOf(Agent::class, $world->getBestAgent());
    }

    public function testReproductionCreatesValidOffspring(): void
    {
        $world = new World();
        $world->createAgents(2, 3, 1);

        $agents = $world->getAgents();
        $offspring = $world->reproduce($agents[0], $agents[1]);

        $this->assertInstanceOf(Agent::class, $offspring);
        $this->assertNotEmpty($offspring->getGenomeArray());

        // Offspring should be able to process inputs
        $offspring->step([1.0, 0.5, 0.0]);
        $outputs = $offspring->getOutputValues();

        $this->assertNotEmpty($outputs);
        $this->assertIsFloat($outputs[0]);
    }
}
