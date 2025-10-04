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
    define('PROBABILITY_MUTATE_ADD_NEURON', 0);
}
if (!defined('PROBABILITY_MUTATE_REMOVE_NEURON')) {
    define('PROBABILITY_MUTATE_REMOVE_NEURON', 0);
}
if (!defined('PROBABILITY_MUTATE_ADD_GENE')) {
    define('PROBABILITY_MUTATE_ADD_GENE', 0);
}
if (!defined('PROBABILITY_MUTATE_REMOVE_GENE')) {
    define('PROBABILITY_MUTATE_REMOVE_GENE', 0);
}
if (!defined('SAVE_WORLD_EVERY_GENERATION')) {
    define('SAVE_WORLD_EVERY_GENERATION', 0);
}

class AutoEncoderTest extends TestCase
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

    public function testAutoEncoderBasic(): void
    {
        $population = 40;
        $generations = 25;

        // AutoEncoder: Input should equal Output (compression through bottleneck)
        // 4 inputs -> hidden layers -> 4 outputs
        $data = [
            [[1, 1, 0, 0, 0], [1, 0, 0, 0]],
            [[1, 0, 1, 0, 0], [0, 1, 0, 0]],
            [[1, 0, 0, 1, 0], [0, 0, 1, 0]],
            [[1, 0, 0, 0, 1], [0, 0, 0, 1]],
        ];

        $fitnessFunction = function (Agent $agent, $dataRow) {
            $predictedOutputs = $agent->getOutputValues();
            $actualOutputs = $dataRow[1];

            $error = 0;
            for ($i = 0; $i < count($actualOutputs); $i++) {
                $error += abs($predictedOutputs[$i] - $actualOutputs[$i]);
            }

            return count($actualOutputs) - $error;
        };

        $world = new World('autoencoder_test');
        $world->createAgents($population, 5, 4, [3]); // 5 inputs (with bias), 4 outputs, 3 hidden (bottleneck)

        $world->step($fitnessFunction, $data, $generations, 0.3);

        $bestAgent = $world->getBestAgent();

        $this->assertInstanceOf(Agent::class, $bestAgent);
        $this->assertGreaterThan(0, $bestAgent->getFitness());

        // Test reconstruction
        $bestAgent->reset();
        $testInput = [1, 1, 0, 0, 0]; // First pattern
        $bestAgent->step($testInput);
        $outputs = $bestAgent->getOutputValues();

        $this->assertCount(4, $outputs);
    }

    public function testAutoEncoderReconstruction(): void
    {
        $population = 30;
        $generations = 20;

        // Simple 2-bit patterns
        $data = [
            [[1, 1, 0], [1, 0]],
            [[1, 0, 1], [0, 1]],
        ];

        $fitnessFunction = function (Agent $agent, $dataRow) {
            $predictedOutputs = $agent->getOutputValues();
            $actualOutputs = $dataRow[1];

            $error = 0;
            for ($i = 0; $i < count($actualOutputs); $i++) {
                $error += abs($predictedOutputs[$i] - $actualOutputs[$i]);
            }

            return count($actualOutputs) - $error;
        };

        $world = new World();
        $world->createAgents($population, 3, 2, [2]); // Bottleneck of 2 neurons

        $world->step($fitnessFunction, $data, $generations, 0.3);

        $bestAgent = $world->getBestAgent();
        $bestAgent->reset();

        $totalError = 0;

        foreach ($data as $row) {
            $bestAgent->step($row[0]);
            $outputs = $bestAgent->getOutputValues();

            for ($i = 0; $i < count($row[1]); $i++) {
                $totalError += abs($outputs[$i] - $row[1][$i]);
            }
        }

        $averageError = $totalError / (count($data) * 2);

        // Should learn to reconstruct with reasonable accuracy
        $this->assertLessThan(0.5, $averageError);
    }

    public function testAutoEncoderWithMultipleOutputs(): void
    {
        $population = 25;
        $generations = 15;

        $data = [
            [[1, 1, 0, 0], [1, 0, 0]],
            [[1, 0, 1, 0], [0, 1, 0]],
            [[1, 0, 0, 1], [0, 0, 1]],
        ];

        $fitnessFunction = function (Agent $agent, $dataRow) {
            $predictedOutputs = $agent->getOutputValues();
            $actualOutputs = $dataRow[1];

            $fitness = 0;
            for ($i = 0; $i < count($actualOutputs); $i++) {
                $fitness += (1.0 - abs($predictedOutputs[$i] - $actualOutputs[$i]));
            }

            return $fitness;
        };

        $world = new World();
        $world->createAgents($population, 4, 3, [2]);

        $world->step($fitnessFunction, $data, $generations, 0.3);

        $this->assertInstanceOf(Agent::class, $world->getBestAgent());
        $this->assertGreaterThan(0, $world->getBestAgent()->getFitness());
    }

    public function testBottleneckCompression(): void
    {
        // Test that network can compress information through bottleneck
        $agent = new Agent();
        $agent->createNeuron(\Rotifer\Models\Neuron::TYPE_INPUT, 4);
        $agent->createNeuron(\Rotifer\Models\Neuron::TYPE_HIDDEN, 2); // Bottleneck
        $agent->createNeuron(\Rotifer\Models\Neuron::TYPE_OUTPUT, 4);
        $agent->initRandomConnections();

        // Bottleneck forces compression
        $hiddenNeurons = $agent->getNeuronsByType(\Rotifer\Models\Neuron::TYPE_HIDDEN);
        $inputNeurons = $agent->getNeuronsByType(\Rotifer\Models\Neuron::TYPE_INPUT);

        $this->assertLessThan(count($inputNeurons), count($hiddenNeurons));
    }

    public function testAutoEncoderDifferentArchitectures(): void
    {
        $data = [
            [[1, 1, 0], [1, 0]],
            [[1, 0, 1], [0, 1]],
        ];

        $fitnessFunction = function (Agent $agent, $dataRow) {
            $predictedOutputs = $agent->getOutputValues();
            $actualOutputs = $dataRow[1];

            $fitness = 0;
            for ($i = 0; $i < count($actualOutputs); $i++) {
                $fitness += (1.0 - abs($predictedOutputs[$i] - $actualOutputs[$i]));
            }

            return $fitness;
        };

        $architectures = [
            [2],      // Single bottleneck layer
            [3, 2],   // Two layers with bottleneck
            [2, 3],   // Expanding layers
        ];

        foreach ($architectures as $layers) {
            $world = new World();
            $world->createAgents(20, 3, 2, $layers);
            $world->step($fitnessFunction, $data, 10, 0.3);

            $this->assertInstanceOf(Agent::class, $world->getBestAgent());
            $this->assertGreaterThan(0, $world->getBestAgent()->getFitness());
        }
    }

    public function testAutoEncoderFitnessImprovement(): void
    {
        $population = 30;

        $data = [
            [[1, 1, 0, 0], [1, 0, 0]],
            [[1, 0, 1, 0], [0, 1, 0]],
            [[1, 0, 0, 1], [0, 0, 1]],
        ];

        $fitnessFunction = function (Agent $agent, $dataRow) {
            $predictedOutputs = $agent->getOutputValues();
            $actualOutputs = $dataRow[1];

            $fitness = 0;
            for ($i = 0; $i < count($actualOutputs); $i++) {
                $fitness += (1.0 - abs($predictedOutputs[$i] - $actualOutputs[$i]));
            }

            return $fitness;
        };

        $world = new World();
        $world->createAgents($population, 4, 3, [2]);

        // First generation
        $world->nextGeneration($fitnessFunction, $data, 0.3);
        $firstGenFitness = $world->getBestAgent()->getFitness();

        // 15 more generations
        for ($i = 0; $i < 15; $i++) {
            $world->nextGeneration($fitnessFunction, $data, 0.3);
        }

        $finalFitness = $world->getBestAgent()->getFitness();

        // Fitness should improve
        $this->assertGreaterThanOrEqual($firstGenFitness, $finalFitness);
    }

    public function testAutoEncoderIdentityMapping(): void
    {
        // Test if network can learn identity mapping (input = output)
        $population = 25;

        $data = [
            [[1, 0.8], [0.8]],
            [[1, 0.2], [0.2]],
            [[1, 0.5], [0.5]],
        ];

        $fitnessFunction = function (Agent $agent, $dataRow) {
            $predictedOutput = $agent->getOutputValues()[0];
            $actualOutput = $dataRow[1][0];
            return (1.0 - abs($predictedOutput - $actualOutput));
        };

        $world = new World();
        $world->createAgents($population, 2, 1, [2]);

        $world->step($fitnessFunction, $data, 20, 0.3);

        $bestAgent = $world->getBestAgent();
        $bestAgent->reset();

        $errors = [];
        foreach ($data as $row) {
            $bestAgent->step($row[0]);
            $predicted = $bestAgent->getOutputValues()[0];
            $actual = $row[1][0];
            $errors[] = abs($predicted - $actual);
        }

        $averageError = array_sum($errors) / count($errors);

        // Should learn identity mapping with good accuracy
        $this->assertLessThan(0.4, $averageError);
    }
}
