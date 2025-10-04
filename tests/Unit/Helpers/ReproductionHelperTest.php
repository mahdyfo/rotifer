<?php

namespace Rotifer\Tests\Unit\Helpers;

use PHPUnit\Framework\TestCase;
use Rotifer\Helpers\ReproductionHelper;
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
if (!defined('MUTATE_WEIGHT_COUNT')) {
    define('MUTATE_WEIGHT_COUNT', 1);
}

class ReproductionHelperTest extends TestCase
{
    private function createSimpleAgent(bool $hasMemory = false): Agent
    {
        $agent = new Agent();
        $agent->setHasMemory($hasMemory);
        $agent->createNeuron(Neuron::TYPE_INPUT, 2);
        $agent->createNeuron(Neuron::TYPE_HIDDEN, 2);
        $agent->createNeuron(Neuron::TYPE_OUTPUT, 1);
        $agent->initRandomConnections();

        return $agent;
    }

    public function testCrossoverReturnsAgent(): void
    {
        $agent1 = $this->createSimpleAgent();
        $agent2 = $this->createSimpleAgent();

        $offspring = ReproductionHelper::crossover($agent1, $agent2, 0.5);

        $this->assertInstanceOf(Agent::class, $offspring);
    }

    public function testCrossoverMemoryInheritance(): void
    {
        // Both parents have memory
        $agent1 = $this->createSimpleAgent(true);
        $agent2 = $this->createSimpleAgent(true);

        $offspring = ReproductionHelper::crossover($agent1, $agent2, 0.5);
        $this->assertTrue($offspring->hasMemory());
    }

    public function testCrossoverNoMemoryInheritance(): void
    {
        // Neither parent has memory
        $agent1 = $this->createSimpleAgent(false);
        $agent2 = $this->createSimpleAgent(false);

        $offspring = ReproductionHelper::crossover($agent1, $agent2, 0.5);
        $this->assertFalse($offspring->hasMemory());
    }

    public function testCrossoverWithZeroProbability(): void
    {
        $agent1 = $this->createSimpleAgent();
        $agent2 = $this->createSimpleAgent();

        $genome1 = $agent1->getGenomeArray();

        $offspring = ReproductionHelper::crossover($agent1, $agent2, 0.0);
        $offspringGenome = $offspring->getGenomeArray();

        // With 0 probability, offspring should be identical to agent1
        $this->assertCount(count($genome1), $offspringGenome);
    }

    public function testCrossoverProducesValidGenome(): void
    {
        $agent1 = $this->createSimpleAgent();
        $agent2 = $this->createSimpleAgent();

        $offspring = ReproductionHelper::crossover($agent1, $agent2, 0.5);
        $genome = $offspring->getGenomeArray();

        $this->assertNotEmpty($genome);

        // Verify genome structure
        foreach ($genome as $gene) {
            $this->assertArrayHasKey('from_type', $gene);
            $this->assertArrayHasKey('from_index', $gene);
            $this->assertArrayHasKey('to_type', $gene);
            $this->assertArrayHasKey('to_index', $gene);
            $this->assertArrayHasKey('weight', $gene);
        }
    }

    public function testMutateReturnsAgent(): void
    {
        $agent = $this->createSimpleAgent();

        $mutated = ReproductionHelper::mutate($agent, 0.5, 0.1, 0.1, 0.1, 0.1);

        $this->assertInstanceOf(Agent::class, $mutated);
    }

    public function testMutateChangeWeight(): void
    {
        $agent = $this->createSimpleAgent();
        $originalGenome = $agent->getGenomeArray();

        // Force weight mutation with 100% probability
        $mutated = ReproductionHelper::mutate($agent, 1.0, 0.0, 0.0, 0.0, 0.0);
        $mutatedGenome = $mutated->getGenomeArray();

        // At least one weight should have changed
        $weightChanged = false;
        for ($i = 0; $i < min(count($originalGenome), count($mutatedGenome)); $i++) {
            if (isset($originalGenome[$i]) && isset($mutatedGenome[$i])) {
                if ($originalGenome[$i]['weight'] !== $mutatedGenome[$i]['weight']) {
                    $weightChanged = true;
                    break;
                }
            }
        }

        $this->assertTrue($weightChanged);
    }

    public function testMutateAddNeuron(): void
    {
        $agent = $this->createSimpleAgent();
        $originalHiddenCount = count($agent->getNeuronsByType(Neuron::TYPE_HIDDEN));

        // Try to add neuron (may not always succeed due to randomness)
        $attempts = 0;
        $neuronAdded = false;

        while ($attempts < 100 && !$neuronAdded) {
            $testAgent = $this->createSimpleAgent();
            $mutated = ReproductionHelper::mutate($testAgent, 0.0, 1.0, 0.0, 0.0, 0.0);
            $newHiddenCount = count($mutated->getNeuronsByType(Neuron::TYPE_HIDDEN));

            if ($newHiddenCount > $originalHiddenCount) {
                $neuronAdded = true;
            }

            $attempts++;
        }

        // With 100% probability and 100 attempts, we should add at least one neuron
        $this->assertTrue($neuronAdded);
    }

    public function testMutateAddConnection(): void
    {
        $agent = $this->createSimpleAgent();
        $originalGenomeSize = count($agent->getGenomeArray());

        // Try to add connection
        $attempts = 0;
        $connectionAdded = false;

        while ($attempts < 100 && !$connectionAdded) {
            $testAgent = $this->createSimpleAgent();
            $mutated = ReproductionHelper::mutate($testAgent, 0.0, 0.0, 1.0, 0.0, 0.0);
            $newGenomeSize = count($mutated->getGenomeArray());

            if ($newGenomeSize > $originalGenomeSize) {
                $connectionAdded = true;
            }

            $attempts++;
        }

        // Should be able to add connection with multiple attempts
        $this->assertTrue($connectionAdded);
    }

    public function testMutateDeleteNeuron(): void
    {
        // Create agent with multiple hidden neurons
        $agent = new Agent();
        $agent->createNeuron(Neuron::TYPE_INPUT, 2);
        $agent->createNeuron(Neuron::TYPE_HIDDEN, 5);
        $agent->createNeuron(Neuron::TYPE_OUTPUT, 1);
        $agent->initRandomConnections();

        $originalHiddenCount = count($agent->getNeuronsByType(Neuron::TYPE_HIDDEN));

        // Try to delete neuron
        $attempts = 0;
        $neuronDeleted = false;

        while ($attempts < 100 && !$neuronDeleted) {
            $testAgent = new Agent();
            $testAgent->createNeuron(Neuron::TYPE_INPUT, 2);
            $testAgent->createNeuron(Neuron::TYPE_HIDDEN, 5);
            $testAgent->createNeuron(Neuron::TYPE_OUTPUT, 1);
            $testAgent->initRandomConnections();

            $mutated = ReproductionHelper::mutate($testAgent, 0.0, 0.0, 0.0, 1.0, 0.0);
            $newHiddenCount = count($mutated->getNeuronsByType(Neuron::TYPE_HIDDEN));

            if ($newHiddenCount < $originalHiddenCount) {
                $neuronDeleted = true;
            }

            $attempts++;
        }

        $this->assertTrue($neuronDeleted);
    }

    public function testMutateDeleteConnection(): void
    {
        $agent = $this->createSimpleAgent();
        $originalGenomeSize = count($agent->getGenomeArray());

        // Try to delete connection
        $attempts = 0;
        $connectionDeleted = false;

        while ($attempts < 100 && !$connectionDeleted) {
            $testAgent = $this->createSimpleAgent();
            $mutated = ReproductionHelper::mutate($testAgent, 0.0, 0.0, 0.0, 0.0, 1.0);
            $newGenomeSize = count($mutated->getGenomeArray());

            if ($newGenomeSize < $originalGenomeSize) {
                $connectionDeleted = true;
            }

            $attempts++;
        }

        $this->assertTrue($connectionDeleted);
    }

    public function testMutateWithZeroProbabilities(): void
    {
        $agent = $this->createSimpleAgent();
        $originalGenome = $agent->getGenomeArray();

        $mutated = ReproductionHelper::mutate($agent, 0.0, 0.0, 0.0, 0.0, 0.0);
        $mutatedGenome = $mutated->getGenomeArray();

        // With zero probabilities, genome should remain similar
        $this->assertCount(count($originalGenome), $mutatedGenome);
    }

    public function testMutatedAgentIsValid(): void
    {
        $agent = $this->createSimpleAgent();

        $mutated = ReproductionHelper::mutate($agent, 0.5, 0.3, 0.3, 0.1, 0.1);

        // Mutated agent should be able to step
        $mutated->step([1.0, 0.5]);
        $outputs = $mutated->getOutputValues();

        $this->assertNotEmpty($outputs);
        $this->assertIsFloat($outputs[0]);
    }

    public function testCrossoverAndMutationPreserveInputOutputCounts(): void
    {
        $agent1 = $this->createSimpleAgent();
        $agent2 = $this->createSimpleAgent();

        $offspring = ReproductionHelper::crossover($agent1, $agent2, 0.5);
        $mutated = ReproductionHelper::mutate($offspring, 0.5, 0.3, 0.3, 0.1, 0.1);

        // Input and output counts should remain the same
        $this->assertCount(2, $mutated->getNeuronsByType(Neuron::TYPE_INPUT));
        $this->assertCount(1, $mutated->getNeuronsByType(Neuron::TYPE_OUTPUT));
    }

    public function testMultipleMutationsInSequence(): void
    {
        $agent = $this->createSimpleAgent();

        // Apply multiple mutations
        for ($i = 0; $i < 5; $i++) {
            $agent = ReproductionHelper::mutate($agent, 0.3, 0.1, 0.1, 0.05, 0.05);
        }

        // Agent should still be valid
        $this->assertInstanceOf(Agent::class, $agent);
        $this->assertNotEmpty($agent->getGenomeArray());

        // Should still be able to process inputs
        $agent->step([1.0, 0.5]);
        $outputs = $agent->getOutputValues();
        $this->assertNotEmpty($outputs);
    }
}
