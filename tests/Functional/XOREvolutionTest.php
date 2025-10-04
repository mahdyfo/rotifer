<?php

namespace Rotifer\Tests\Functional;

use PHPUnit\Framework\TestCase;
use Rotifer\Models\World;
use Rotifer\Models\Agent;
use Rotifer\Models\StaticAgent;
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

class XOREvolutionTest extends TestCase
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

    public function testXORWithStaticAgents(): void
    {
        $population = 100;
        $generations = 50;
        $layers = [4, 3];

        $data = [
            [[1, 0, 0], [0]],
            [[1, 0, 1], [1]],
            [[1, 1, 0], [1]],
            [[1, 1, 1], [0]],
        ];

        $fitnessFunction = function (StaticAgent $agent, $dataRow) {
            $predictedOutput = $agent->getOutputValues()[0];
            $actualOutput = $dataRow[1][0];
            return (1.0 - abs($predictedOutput - $actualOutput));
        };

        $world = new World('xor_test');
        $world->createAgents($population, 3, 1, $layers);
        $world->step($fitnessFunction, $data, $generations, 0.2);

        $bestAgent = $world->getBestAgent();

        // Test the best agent
        $this->assertInstanceOf(Agent::class, $bestAgent);
        $this->assertGreaterThan(0, $bestAgent->getFitness());

        $bestAgent->reset();
        $errors = [];

        foreach ($data as $row) {
            $bestAgent->step($row[0]);
            $predicted = $bestAgent->getOutputValues()[0];
            $actual = $row[1][0];
            $errors[] = abs($predicted - $actual);
        }

        $averageError = array_sum($errors) / count($errors);

        // Should show learning progress (average error less than 0.5)
        // Note: XOR is a hard problem, evolution may not always fully converge
        $this->assertLessThan(0.5, $averageError);
    }

    public function testXORWithDynamicAgents(): void
    {
        $population = 50;
        $generations = 40;

        $data = [
            [[1, 0, 0], [0]],
            [[1, 0, 1], [1]],
            [[1, 1, 0], [1]],
            [[1, 1, 1], [0]],
        ];

        $fitnessFunction = function (Agent $agent, $dataRow) {
            $predictedOutput = $agent->getOutputValues()[0];
            $actualOutput = $dataRow[1][0];
            return (1.0 - abs($predictedOutput - $actualOutput));
        };

        $world = new World('xor_dynamic_test');
        $world->createAgents($population, 3, 1); // No layers = dynamic
        $world->step($fitnessFunction, $data, $generations, 0.2);

        $bestAgent = $world->getBestAgent();

        $this->assertInstanceOf(Agent::class, $bestAgent);
        $this->assertGreaterThan(0, $bestAgent->getFitness());

        // Network should have evolved some hidden neurons
        $hiddenNeurons = $bestAgent->getNeuronsByType(\Rotifer\Models\Neuron::TYPE_HIDDEN);
        $this->assertGreaterThan(0, count($hiddenNeurons));
    }

    public function testXORFitnessImprovement(): void
    {
        $population = 30;

        $data = [
            [[1, 0, 0], [0]],
            [[1, 0, 1], [1]],
            [[1, 1, 0], [1]],
            [[1, 1, 1], [0]],
        ];

        $fitnessFunction = function (Agent $agent, $dataRow) {
            $predictedOutput = $agent->getOutputValues()[0];
            $actualOutput = $dataRow[1][0];
            return (1.0 - abs($predictedOutput - $actualOutput));
        };

        $world = new World();
        $world->createAgents($population, 3, 1, [3, 2]);

        // First generation
        $world->nextGeneration($fitnessFunction, $data, 0.2);
        $firstGenFitness = $world->getBestAgent()->getFitness();

        // 20 more generations
        for ($i = 0; $i < 20; $i++) {
            $world->nextGeneration($fitnessFunction, $data, 0.2);
        }

        $finalFitness = $world->getBestAgent()->getFitness();

        // Fitness should improve or stay high (genetic algorithms are stochastic)
        // Allow some tolerance since initial generation might be lucky
        $this->assertGreaterThanOrEqual($firstGenFitness * 0.9, $finalFitness,
            "Fitness degraded too much. First gen: {$firstGenFitness}, Final: {$finalFitness}");
    }

    public function testXORPredictionAccuracy(): void
    {
        $world = new World();
        $world->createAgents(40, 3, 1, [4]);

        $data = [
            [[1, 0, 0], [0]],
            [[1, 0, 1], [1]],
            [[1, 1, 0], [1]],
            [[1, 1, 1], [0]],
        ];

        $fitnessFunction = function (Agent $agent, $dataRow) {
            $predictedOutput = $agent->getOutputValues()[0];
            $actualOutput = $dataRow[1][0];
            return (1.0 - abs($predictedOutput - $actualOutput));
        };

        $world->step($fitnessFunction, $data, 35, 0.2);

        $bestAgent = $world->getBestAgent();
        $bestAgent->reset();

        $correctPredictions = 0;

        foreach ($data as $row) {
            $bestAgent->step($row[0]);
            $predicted = round($bestAgent->getOutputValues()[0]);
            $actual = $row[1][0];

            if ($predicted == $actual) {
                $correctPredictions++;
            }
        }

        // Should get at least 2 out of 4 correct (50%)
        $this->assertGreaterThanOrEqual(2, $correctPredictions);
    }

    public function testXORWithDifferentSurvivalRates(): void
    {
        $data = [
            [[1, 0, 0], [0]],
            [[1, 0, 1], [1]],
            [[1, 1, 0], [1]],
            [[1, 1, 1], [0]],
        ];

        $fitnessFunction = function (Agent $agent, $dataRow) {
            $predictedOutput = $agent->getOutputValues()[0];
            $actualOutput = $dataRow[1][0];
            return (1.0 - abs($predictedOutput - $actualOutput));
        };

        $survivalRates = [0.2, 0.5, 0.8];

        foreach ($survivalRates as $survivalRate) {
            $world = new World();
            $world->createAgents(20, 3, 1, [3]);
            $world->step($fitnessFunction, $data, 15, $survivalRate);

            $this->assertInstanceOf(Agent::class, $world->getBestAgent());
            $this->assertGreaterThan(0, $world->getBestAgent()->getFitness());
        }
    }

    public function testXORConvergence(): void
    {
        $world = new World();
        $world->createAgents(60, 3, 1, [4, 3]);

        $data = [
            [[1, 0, 0], [0]],
            [[1, 0, 1], [1]],
            [[1, 1, 0], [1]],
            [[1, 1, 1], [0]],
        ];

        $fitnessFunction = function (Agent $agent, $dataRow) {
            $predictedOutput = $agent->getOutputValues()[0];
            $actualOutput = $dataRow[1][0];
            return (1.0 - abs($predictedOutput - $actualOutput));
        };

        $stopFunction = function (World $world) {
            return $world->getBestAgent()->getFitness() >= 3.0; // Good fitness (75% accuracy)
        };

        $world->step($fitnessFunction, $data, 100, 0.2, 0, $stopFunction);

        // Should reach good fitness (genetic algorithms are stochastic)
        $this->assertGreaterThanOrEqual(3.0, $world->getBestAgent()->getFitness());
    }
}
