<?php

namespace Rotifer\Tests\Unit\Models;

use PHPUnit\Framework\TestCase;
use Rotifer\Models\World;
use Rotifer\Models\Agent;
use Rotifer\Models\Neuron;
use Rotifer\Activations\Activation;

// Define constants for testing
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

class WorldTest extends TestCase
{
    protected function setUp(): void
    {
        parent::setUp();

        // Clean up autosave directory before each test
        if (is_dir('autosave')) {
            $files = glob('autosave/*');
            foreach ($files as $file) {
                if (is_file($file)) {
                    unlink($file);
                }
            }
        }
    }

    protected function tearDown(): void
    {
        parent::tearDown();

        // Clean up autosave directory after each test
        if (is_dir('autosave')) {
            $files = glob('autosave/*');
            foreach ($files as $file) {
                if (is_file($file)) {
                    unlink($file);
                }
            }
        }
    }

    public function testWorldCreation(): void
    {
        $world = new World('test_world');
        $this->assertInstanceOf(World::class, $world);
    }

    public function testCreateAgentsThrowsExceptionWithOneAgent(): void
    {
        $this->expectException(\Exception::class);
        $this->expectExceptionMessage('The world cannot have only 1 agent');

        $world = new World();
        $world->createAgents(1, 2, 1);
    }

    public function testCreateDynamicAgents(): void
    {
        $world = new World();
        $world->createAgents(10, 3, 2); // Dynamic agents (no layers specified)

        $agents = $world->getAgents();

        $this->assertCount(10, $agents);

        foreach ($agents as $agent) {
            $this->assertInstanceOf(Agent::class, $agent);
            $this->assertCount(3, $agent->getNeuronsByType(Neuron::TYPE_INPUT));
            $this->assertCount(2, $agent->getNeuronsByType(Neuron::TYPE_OUTPUT));
            $this->assertNotEmpty($agent->getNeuronsByType(Neuron::TYPE_HIDDEN));
        }
    }

    public function testCreateStaticAgents(): void
    {
        $world = new World();
        $world->createAgents(10, 3, 2, [4, 3]); // Static agents with 2 layers

        $agents = $world->getAgents();

        $this->assertCount(10, $agents);

        foreach ($agents as $agent) {
            $this->assertInstanceOf(Agent::class, $agent);
            $this->assertCount(3, $agent->getNeuronsByType(Neuron::TYPE_INPUT));
            $this->assertCount(2, $agent->getNeuronsByType(Neuron::TYPE_OUTPUT));
            $this->assertCount(7, $agent->getNeuronsByType(Neuron::TYPE_HIDDEN)); // 4 + 3 = 7
        }
    }

    public function testSetAndGetAgents(): void
    {
        $world = new World();
        $agents = [];

        for ($i = 0; $i < 5; $i++) {
            $agent = new Agent();
            $agent->createNeuron(Neuron::TYPE_INPUT, 2);
            $agent->createNeuron(Neuron::TYPE_OUTPUT, 1);
            $agents[] = $agent;
        }

        $world->setAgents($agents);
        $retrievedAgents = $world->getAgents();

        $this->assertCount(5, $retrievedAgents);
        $this->assertEquals($agents, $retrievedAgents);
    }

    public function testReproduce(): void
    {
        $world = new World();
        $world->createAgents(2, 3, 1);

        $agents = $world->getAgents();
        $childAgent = $world->reproduce($agents[0], $agents[1]);

        $this->assertInstanceOf(Agent::class, $childAgent);
        $this->assertNotEmpty($childAgent->getGenomeArray());
    }

    public function testTournamentReproduction(): void
    {
        $world = new World();
        $world->createAgents(20, 3, 1);

        $agentAndFitnessArray = [];
        foreach ($world->getAgents() as $key => $agent) {
            $agentAndFitnessArray[] = [
                'fitness' => mt_rand(0, 100) / 10,
                'agent_key' => $key
            ];
        }

        $newAgents = $world->tournament($agentAndFitnessArray);

        $this->assertCount(20, $newAgents);

        foreach ($newAgents as $agent) {
            $this->assertInstanceOf(Agent::class, $agent);
        }
    }

    public function testNextGeneration(): void
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

        $world->nextGeneration($fitnessFunction, $data, 0.2);

        // Verify best agent is set
        $bestAgent = $world->getBestAgent();
        $this->assertInstanceOf(Agent::class, $bestAgent);
        $this->assertGreaterThan(0, $bestAgent->getFitness());
    }

    public function testStepMultipleGenerations(): void
    {
        $world = new World();
        $world->createAgents(10, 3, 1);

        $data = [
            [[1, 0, 0], [0]],
            [[1, 0, 1], [1]],
        ];

        $fitnessFunction = function (Agent $agent, $dataRow) {
            $predicted = $agent->getOutputValues()[0];
            $actual = $dataRow[1][0];
            return 1.0 - abs($predicted - $actual);
        };

        $world->step($fitnessFunction, $data, 3, 0.5);

        $bestAgent = $world->getBestAgent();
        $this->assertInstanceOf(Agent::class, $bestAgent);
    }

    public function testStopFunction(): void
    {
        $world = new World();
        $world->createAgents(10, 3, 1);

        $data = [
            [[1, 0, 0], [0]],
        ];

        $fitnessFunction = function (Agent $agent, $dataRow) {
            return 1.0;
        };

        $stopFunction = function (World $world) {
            return $world->getBestAgent()->getFitness() >= 0.9;
        };

        $world->step($fitnessFunction, $data, 100, 0.5, 0, $stopFunction);

        $this->assertGreaterThanOrEqual(0.9, $world->getBestAgent()->getFitness());
    }

    public function testGetBestAgent(): void
    {
        $world = new World();
        $world->createAgents(5, 3, 1);

        $data = [[[1, 0, 0], [0]]];

        $fitnessFunction = function (Agent $agent) {
            return mt_rand(0, 100) / 10;
        };

        $world->nextGeneration($fitnessFunction, $data, 0.5);

        $bestAgent = $world->getBestAgent();
        $this->assertInstanceOf(Agent::class, $bestAgent);

        // Best agent should have the highest fitness
        foreach ($world->getAgents() as $agent) {
            $this->assertLessThanOrEqual($bestAgent->getFitness(), $agent->getFitness());
        }
    }

    public function testMemoryReset(): void
    {
        $world = new World();
        $world->createAgents(5, 3, 1, [], true); // hasMemory = true

        $data = [
            [[1, 0, 0], [0]],
            [], // Empty row to reset memory
            [[1, 1, 1], [1]],
        ];

        $fitnessFunction = function (Agent $agent, $dataRow) {
            if (empty($dataRow)) {
                return 0;
            }
            return 1.0;
        };

        $world->nextGeneration($fitnessFunction, $data, 0.5);

        // Test should complete without errors
        $this->assertInstanceOf(Agent::class, $world->getBestAgent());
    }

    public function testAgentsHaveGenomes(): void
    {
        $world = new World();
        $world->createAgents(10, 3, 1);

        foreach ($world->getAgents() as $agent) {
            $genome = $agent->getGenomeArray();
            $this->assertNotEmpty($genome);
        }
    }

    public function testWorldWithMemoryAgents(): void
    {
        $world = new World();
        $world->createAgents(10, 3, 1, [], true);

        foreach ($world->getAgents() as $agent) {
            $this->assertTrue($agent->hasMemory());
        }
    }

    public function testWorldWithoutMemoryAgents(): void
    {
        $world = new World();
        $world->createAgents(10, 3, 1, [], false);

        foreach ($world->getAgents() as $agent) {
            $this->assertFalse($agent->hasMemory());
        }
    }
}
