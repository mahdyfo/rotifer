<?php

namespace Rotifer\Tests\Functional;

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

class MemoryNetworkTest extends TestCase
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

    public function testAgentMemoryPersistence(): void
    {
        $agent = new Agent();
        $agent->setHasMemory(true);
        $agent->createNeuron(\Rotifer\Models\Neuron::TYPE_INPUT, 2);
        $agent->createNeuron(\Rotifer\Models\Neuron::TYPE_HIDDEN, 3);
        $agent->createNeuron(\Rotifer\Models\Neuron::TYPE_OUTPUT, 1);

        // Create a network with self-connections to ensure memory effect
        $genome = [
            ['from_type' => 0, 'from_index' => 0, 'to_type' => 1, 'to_index' => 0, 'weight' => 1.0],
            ['from_type' => 1, 'from_index' => 0, 'to_type' => 1, 'to_index' => 0, 'weight' => 0.5], // self-connection
            ['from_type' => 1, 'from_index' => 0, 'to_type' => 2, 'to_index' => 0, 'weight' => 1.0],
        ];
        $agent->setGenome($genome);

        // First step
        $agent->step([1.0, 0.5]);
        $firstOutputs = $agent->getOutputValues();

        // Second step with same inputs
        $agent->step([1.0, 0.5]);
        $secondOutputs = $agent->getOutputValues();

        // Outputs should be different due to memory (self-connection accumulates)
        $this->assertNotSame($firstOutputs[0], $secondOutputs[0]);
    }

    public function testAgentWithoutMemory(): void
    {
        $agent = new Agent();
        $agent->setHasMemory(false);
        $agent->createNeuron(\Rotifer\Models\Neuron::TYPE_INPUT, 2);
        $agent->createNeuron(\Rotifer\Models\Neuron::TYPE_HIDDEN, 3);
        $agent->createNeuron(\Rotifer\Models\Neuron::TYPE_OUTPUT, 1);
        $agent->initRandomConnections();

        // First step
        $agent->step([1.0, 0.5]);
        $firstOutputs = $agent->getOutputValues();

        // Second step with same inputs
        $agent->step([1.0, 0.5]);
        $secondOutputs = $agent->getOutputValues();

        // Outputs should be the same without memory
        $this->assertEqualsWithDelta($firstOutputs[0], $secondOutputs[0], 0.0001);
    }

    public function testMemoryReset(): void
    {
        $agent = new Agent();
        $agent->setHasMemory(true);
        $agent->createNeuron(\Rotifer\Models\Neuron::TYPE_INPUT, 2);
        $agent->createNeuron(\Rotifer\Models\Neuron::TYPE_HIDDEN, 3);
        $agent->createNeuron(\Rotifer\Models\Neuron::TYPE_OUTPUT, 1);
        $agent->initRandomConnections();

        // First sequence
        $agent->step([1.0, 0.0]);
        $agent->step([0.0, 1.0]);
        $firstSequenceOutput = $agent->getOutputValues()[0];

        // Reset and repeat same sequence
        $agent->resetMemory();
        $agent->step([1.0, 0.0]);
        $agent->step([0.0, 1.0]);
        $secondSequenceOutput = $agent->getOutputValues()[0];

        // Should produce same output after reset
        $this->assertEqualsWithDelta($firstSequenceOutput, $secondSequenceOutput, 0.0001);
    }

    public function testSequenceLearning(): void
    {
        $population = 30;
        $generations = 25;

        // Sequence learning: output should depend on previous input
        $data = [
            [[1, 1], [0]], // First in sequence
            [[1, 0], [1]], // After [1,1], output 1
            [[1, 1], [0]], // Restart sequence
            [[1, 0], [1]], // After [1,1], output 1
        ];

        $fitnessFunction = function (Agent $agent, $dataRow) {
            $predictedOutput = $agent->getOutputValues()[0];
            $actualOutput = $dataRow[1][0];
            return (1.0 - abs($predictedOutput - $actualOutput));
        };

        $world = new World('memory_test');
        $world->createAgents($population, 2, 1, [], true); // hasMemory = true

        $world->step($fitnessFunction, $data, $generations, 0.3);

        $bestAgent = $world->getBestAgent();

        $this->assertInstanceOf(Agent::class, $bestAgent);
        $this->assertTrue($bestAgent->hasMemory());
        $this->assertGreaterThan(0, $bestAgent->getFitness());
    }

    public function testMemoryBasedEvolution(): void
    {
        $world = new World();
        $world->createAgents(20, 2, 1, [], true); // Memory agents

        $data = [
            [[1, 0], [0]],
            [[0, 1], [1]],
            [[1, 0], [0]],
        ];

        $fitnessFunction = function (Agent $agent, $dataRow) {
            $predictedOutput = $agent->getOutputValues()[0];
            $actualOutput = $dataRow[1][0];
            return (1.0 - abs($predictedOutput - $actualOutput));
        };

        $world->nextGeneration($fitnessFunction, $data, 0.3);

        foreach ($world->getAgents() as $agent) {
            $this->assertTrue($agent->hasMemory());
        }
    }

    public function testMemoryResetInData(): void
    {
        $world = new World();
        $world->createAgents(20, 2, 1, [], true);

        // Empty row triggers memory reset
        $data = [
            [[1, 0], [0]],
            [[0, 1], [1]],
            [], // Reset memory
            [[1, 0], [0]],
        ];

        $fitnessFunction = function (Agent $agent, $dataRow) {
            if (empty($dataRow)) {
                return 0;
            }
            $predictedOutput = $agent->getOutputValues()[0];
            $actualOutput = $dataRow[1][0];
            return (1.0 - abs($predictedOutput - $actualOutput));
        };

        $world->nextGeneration($fitnessFunction, $data, 0.3);

        $this->assertInstanceOf(Agent::class, $world->getBestAgent());
    }

    public function testLongTermMemory(): void
    {
        $agent = new Agent();
        $agent->setHasMemory(true);
        $agent->createNeuron(\Rotifer\Models\Neuron::TYPE_INPUT, 1);
        $agent->createNeuron(\Rotifer\Models\Neuron::TYPE_HIDDEN, 2);
        $agent->createNeuron(\Rotifer\Models\Neuron::TYPE_OUTPUT, 1);

        // Create network with recurrent connections to ensure memory effect
        $genome = [
            ['from_type' => 0, 'from_index' => 0, 'to_type' => 1, 'to_index' => 0, 'weight' => 1.0],
            ['from_type' => 0, 'from_index' => 0, 'to_type' => 1, 'to_index' => 1, 'weight' => 0.8],
            ['from_type' => 1, 'from_index' => 0, 'to_type' => 1, 'to_index' => 0, 'weight' => 0.3], // self
            ['from_type' => 1, 'from_index' => 0, 'to_type' => 1, 'to_index' => 1, 'weight' => 0.2], // recurrent
            ['from_type' => 1, 'from_index' => 0, 'to_type' => 2, 'to_index' => 0, 'weight' => 1.0],
            ['from_type' => 1, 'from_index' => 1, 'to_type' => 2, 'to_index' => 0, 'weight' => 0.5],
        ];
        $agent->setGenome($genome);

        $outputs = [];

        // Run multiple steps
        for ($i = 0; $i < 10; $i++) {
            $agent->step([1.0]);
            $outputs[] = $agent->getOutputValues()[0];
        }

        // Outputs should change over time due to memory (recurrent connections)
        $uniqueOutputs = array_unique(array_map(function($v) {
            return round($v, 4);
        }, $outputs));

        $this->assertGreaterThan(1, count($uniqueOutputs));
    }

    public function testMemoryInheritance(): void
    {
        $agent1 = new Agent();
        $agent1->setHasMemory(true);
        $agent1->createNeuron(\Rotifer\Models\Neuron::TYPE_INPUT, 2);
        $agent1->createNeuron(\Rotifer\Models\Neuron::TYPE_HIDDEN, 2);
        $agent1->createNeuron(\Rotifer\Models\Neuron::TYPE_OUTPUT, 1);
        $agent1->initRandomConnections();

        $agent2 = new Agent();
        $agent2->setHasMemory(true);
        $agent2->createNeuron(\Rotifer\Models\Neuron::TYPE_INPUT, 2);
        $agent2->createNeuron(\Rotifer\Models\Neuron::TYPE_HIDDEN, 2);
        $agent2->createNeuron(\Rotifer\Models\Neuron::TYPE_OUTPUT, 1);
        $agent2->initRandomConnections();

        $world = new World();
        $offspring = $world->reproduce($agent1, $agent2);

        $this->assertTrue($offspring->hasMemory());
    }

    public function testStepCounter(): void
    {
        $agent = new Agent();
        $agent->setHasMemory(true);
        $agent->createNeuron(\Rotifer\Models\Neuron::TYPE_INPUT, 1);
        $agent->createNeuron(\Rotifer\Models\Neuron::TYPE_OUTPUT, 1);

        $this->assertEquals(0, $agent->getStep());

        $agent->step([1.0]);
        $this->assertEquals(1, $agent->getStep());

        $agent->step([0.5]);
        $this->assertEquals(2, $agent->getStep());

        $agent->resetMemory();
        $this->assertEquals(0, $agent->getStep());
    }
}
