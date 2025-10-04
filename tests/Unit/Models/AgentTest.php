<?php

namespace Rotifer\Tests\Unit\Models;

use PHPUnit\Framework\TestCase;
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

class AgentTest extends TestCase
{
    public function testAgentCreation(): void
    {
        $agent = new Agent();
        $this->assertInstanceOf(Agent::class, $agent);
        $this->assertEquals(0, $agent->getFitness());
    }

    public function testCreateNeurons(): void
    {
        $agent = new Agent();
        $agent->createNeuron(Neuron::TYPE_INPUT, 3);
        $agent->createNeuron(Neuron::TYPE_HIDDEN, 2);
        $agent->createNeuron(Neuron::TYPE_OUTPUT, 1);

        $this->assertCount(3, $agent->getNeuronsByType(Neuron::TYPE_INPUT));
        $this->assertCount(2, $agent->getNeuronsByType(Neuron::TYPE_HIDDEN));
        $this->assertCount(1, $agent->getNeuronsByType(Neuron::TYPE_OUTPUT));
    }

    public function testFindNeuron(): void
    {
        $agent = new Agent();
        $createdNeuron = $agent->createNeuron(Neuron::TYPE_HIDDEN, 1);
        $createdNeuron->setIndex(0);

        $foundNeuron = $agent->findNeuron(Neuron::TYPE_HIDDEN, 0);

        $this->assertNotNull($foundNeuron);
        $this->assertEquals(Neuron::TYPE_HIDDEN, $foundNeuron->getType());
        $this->assertEquals(0, $foundNeuron->getIndex());
    }

    public function testFindNonExistentNeuron(): void
    {
        $agent = new Agent();
        $neuron = $agent->findNeuron(Neuron::TYPE_HIDDEN, 999);

        $this->assertNull($neuron);
    }

    public function testFindOrCreateNeuron(): void
    {
        $agent = new Agent();

        // Create new neuron
        $neuron1 = $agent->findOrCreateNeuron(Neuron::TYPE_HIDDEN, 5);
        $this->assertNotNull($neuron1);
        $this->assertEquals(5, $neuron1->getIndex());

        // Find existing neuron
        $neuron2 = $agent->findOrCreateNeuron(Neuron::TYPE_HIDDEN, 5);
        $this->assertSame($neuron1, $neuron2);
    }

    public function testConnectNeurons(): void
    {
        $agent = new Agent();
        $input = $agent->createNeuron(Neuron::TYPE_INPUT, 1);
        $hidden = $agent->createNeuron(Neuron::TYPE_HIDDEN, 1);

        $agent->connectNeurons($input, $hidden, 1.5);

        $inConnections = $hidden->getInConnections();
        $this->assertEquals(1.5, $inConnections[Neuron::TYPE_INPUT][0]);

        $outConnections = $input->getOutConnections();
        $this->assertEquals(1.5, $outConnections[Neuron::TYPE_HIDDEN][0]);
    }

    public function testCannotConnectInputToInput(): void
    {
        $this->expectException(\Exception::class);
        $this->expectExceptionMessage('Cannot connect input to input');

        $agent = new Agent();
        $input1 = $agent->createNeuron(Neuron::TYPE_INPUT, 1);
        $input2 = $agent->createNeuron(Neuron::TYPE_INPUT, 1);

        $agent->connectNeurons($input1, $input2, 1.0);
    }

    public function testCannotConnectOutputToOutput(): void
    {
        $this->expectException(\Exception::class);
        $this->expectExceptionMessage('Cannot connect output to output');

        $agent = new Agent();
        $output1 = $agent->createNeuron(Neuron::TYPE_OUTPUT, 1);
        $output2 = $agent->createNeuron(Neuron::TYPE_OUTPUT, 1);

        $agent->connectNeurons($output1, $output2, 1.0);
    }

    public function testCannotConnectHiddenToInput(): void
    {
        $this->expectException(\Exception::class);
        $this->expectExceptionMessage('Cannot connect hidden to input');

        $agent = new Agent();
        $hidden = $agent->createNeuron(Neuron::TYPE_HIDDEN, 1);
        $input = $agent->createNeuron(Neuron::TYPE_INPUT, 1);

        $agent->connectNeurons($hidden, $input, 1.0);
    }

    public function testInitRandomConnections(): void
    {
        $agent = new Agent();
        $agent->createNeuron(Neuron::TYPE_INPUT, 2);
        $agent->createNeuron(Neuron::TYPE_HIDDEN, 3);
        $agent->createNeuron(Neuron::TYPE_OUTPUT, 1);

        $agent->initRandomConnections();

        $genome = $agent->getGenomeArray();
        $this->assertNotEmpty($genome);

        // Should have connections from inputs to hiddens and hiddens to outputs
        $hasInputToHidden = false;
        $hasHiddenToOutput = false;

        foreach ($genome as $gene) {
            if ($gene['from_type'] === Neuron::TYPE_INPUT && $gene['to_type'] === Neuron::TYPE_HIDDEN) {
                $hasInputToHidden = true;
            }
            if ($gene['from_type'] === Neuron::TYPE_HIDDEN && $gene['to_type'] === Neuron::TYPE_OUTPUT) {
                $hasHiddenToOutput = true;
            }
        }

        $this->assertTrue($hasInputToHidden);
        $this->assertTrue($hasHiddenToOutput);
    }

    public function testSetAndGetGenome(): void
    {
        $agent = new Agent();
        $agent->createNeuron(Neuron::TYPE_INPUT, 2);
        $agent->createNeuron(Neuron::TYPE_HIDDEN, 1);
        $agent->createNeuron(Neuron::TYPE_OUTPUT, 1);

        $genome = [
            ['from_type' => Neuron::TYPE_INPUT, 'from_index' => 0, 'to_type' => Neuron::TYPE_HIDDEN, 'to_index' => 0, 'weight' => 1.5],
            ['from_type' => Neuron::TYPE_HIDDEN, 'from_index' => 0, 'to_type' => Neuron::TYPE_OUTPUT, 'to_index' => 0, 'weight' => 2.0],
        ];

        $agent->setGenome($genome);
        $retrievedGenome = $agent->getGenomeArray();

        $this->assertCount(2, $retrievedGenome);
        $this->assertEquals(1.5, $retrievedGenome[0]['weight']);
        $this->assertEquals(2.0, $retrievedGenome[1]['weight']);
    }

    public function testStep(): void
    {
        $agent = new Agent();
        $agent->createNeuron(Neuron::TYPE_INPUT, 2);
        $agent->createNeuron(Neuron::TYPE_HIDDEN, 1);
        $agent->createNeuron(Neuron::TYPE_OUTPUT, 1);

        $genome = [
            ['from_type' => Neuron::TYPE_INPUT, 'from_index' => 0, 'to_type' => Neuron::TYPE_HIDDEN, 'to_index' => 0, 'weight' => 1.0],
            ['from_type' => Neuron::TYPE_INPUT, 'from_index' => 1, 'to_type' => Neuron::TYPE_HIDDEN, 'to_index' => 0, 'weight' => 1.0],
            ['from_type' => Neuron::TYPE_HIDDEN, 'from_index' => 0, 'to_type' => Neuron::TYPE_OUTPUT, 'to_index' => 0, 'weight' => 1.0],
        ];

        $agent->setGenome($genome);
        $agent->step([1.0, 0.5]);

        $outputs = $agent->getOutputValues();
        $this->assertCount(1, $outputs);
        $this->assertIsFloat($outputs[0]);
    }

    public function testMemory(): void
    {
        $agent = new Agent();
        $agent->setHasMemory(true);

        $this->assertTrue($agent->hasMemory());

        $agent->setHasMemory(false);
        $this->assertFalse($agent->hasMemory());
    }

    public function testResetMemory(): void
    {
        $agent = new Agent();
        $agent->createNeuron(Neuron::TYPE_HIDDEN, 2);
        $agent->createNeuron(Neuron::TYPE_OUTPUT, 1);

        $hiddens = $agent->getNeuronsByType(Neuron::TYPE_HIDDEN);
        foreach ($hiddens as $neuron) {
            $neuron->setValue(5.0);
        }

        $agent->resetMemory();

        foreach ($hiddens as $neuron) {
            $this->assertEquals(0, $neuron->getValue());
        }
    }

    public function testReset(): void
    {
        $agent = new Agent();
        $agent->setFitness(10.5);
        $agent->createNeuron(Neuron::TYPE_HIDDEN, 1);

        $agent->reset();

        $this->assertEquals(0, $agent->getFitness());
        $this->assertEquals(0, $agent->getStep());
    }

    public function testSetAndGetFitness(): void
    {
        $agent = new Agent();
        $agent->setFitness(7.89);

        $this->assertEquals(7.89, $agent->getFitness());
    }

    public function testGetInputValues(): void
    {
        $agent = new Agent();
        $agent->createNeuron(Neuron::TYPE_INPUT, 3);
        $agent->step([1.0, 2.0, 3.0]);

        $inputs = $agent->getInputValues();

        $this->assertCount(3, $inputs);
        $this->assertEquals([1.0, 2.0, 3.0], $inputs);
    }

    public function testGetOutputValues(): void
    {
        $agent = new Agent();
        $agent->createNeuron(Neuron::TYPE_INPUT, 1);
        $agent->createNeuron(Neuron::TYPE_OUTPUT, 2);

        $genome = [
            ['from_type' => Neuron::TYPE_INPUT, 'from_index' => 0, 'to_type' => Neuron::TYPE_OUTPUT, 'to_index' => 0, 'weight' => 1.0],
            ['from_type' => Neuron::TYPE_INPUT, 'from_index' => 0, 'to_type' => Neuron::TYPE_OUTPUT, 'to_index' => 1, 'weight' => 0.5],
        ];

        $agent->setGenome($genome);
        $agent->step([1.0]);

        $outputs = $agent->getOutputValues();
        $this->assertCount(2, $outputs);
    }

    public function testGetStep(): void
    {
        $agent = new Agent();
        $agent->setHasMemory(true); // Agents with memory don't reset step
        $agent->createNeuron(Neuron::TYPE_INPUT, 1);
        $agent->createNeuron(Neuron::TYPE_OUTPUT, 1);

        $this->assertEquals(0, $agent->getStep());

        $agent->step([1.0]);
        $this->assertEquals(1, $agent->getStep());

        $agent->step([0.5]);
        $this->assertEquals(2, $agent->getStep());
    }

    public function testAdditionalData(): void
    {
        $agent = new Agent();
        $data = ['key1' => 'value1', 'key2' => 42];

        $agent->setAdditional($data);
        $this->assertEquals($data, $agent->getAdditional());
    }

    public function testDeleteRedundantGenes(): void
    {
        $agent = new Agent();
        $agent->createNeuron(Neuron::TYPE_INPUT, 1);
        $agent->createNeuron(Neuron::TYPE_HIDDEN, 2);
        $agent->createNeuron(Neuron::TYPE_OUTPUT, 1);

        // Create a stray hidden neuron with no connections
        $strayNeuron = $agent->createNeuron(Neuron::TYPE_HIDDEN, 1);

        $agent->initRandomConnections();
        $agent->deleteRedundantGenes();

        // The stray neuron should be removed
        $hiddens = $agent->getNeuronsByType(Neuron::TYPE_HIDDEN);
        foreach ($hiddens as $neuron) {
            $this->assertNotEmpty($neuron->getInConnections());
            $this->assertNotEmpty($neuron->getOutConnections());
        }
    }

    public function testRemoveNeuron(): void
    {
        $agent = new Agent();
        $agent->createNeuron(Neuron::TYPE_HIDDEN, 3);

        $hiddens = $agent->getNeuronsByType(Neuron::TYPE_HIDDEN);
        $this->assertCount(3, $hiddens);

        $agent->removeNeuron(Neuron::TYPE_HIDDEN, 1);

        $hiddens = $agent->getNeuronsByType(Neuron::TYPE_HIDDEN);
        $this->assertCount(2, $hiddens);
        $this->assertNull($agent->findNeuron(Neuron::TYPE_HIDDEN, 1));
    }

    public function testGetRandomNeuronByType(): void
    {
        $agent = new Agent();
        $agent->createNeuron(Neuron::TYPE_INPUT, 5);

        $neuron = $agent->getRandomNeuronByType(Neuron::TYPE_INPUT);

        $this->assertInstanceOf(Neuron::class, $neuron);
        $this->assertEquals(Neuron::TYPE_INPUT, $neuron->getType());
    }

    public function testCreateFromGenome(): void
    {
        $genome = [
            ['from_type' => Neuron::TYPE_INPUT, 'from_index' => 0, 'to_type' => Neuron::TYPE_HIDDEN, 'to_index' => 0, 'weight' => 1.5],
            ['from_type' => Neuron::TYPE_HIDDEN, 'from_index' => 0, 'to_type' => Neuron::TYPE_OUTPUT, 'to_index' => 0, 'weight' => 2.0],
        ];

        $agent = Agent::createFromGenome($genome, false);

        $this->assertInstanceOf(Agent::class, $agent);
        $this->assertCount(1, $agent->getNeuronsByType(Neuron::TYPE_INPUT));
        $this->assertCount(1, $agent->getNeuronsByType(Neuron::TYPE_HIDDEN));
        $this->assertCount(1, $agent->getNeuronsByType(Neuron::TYPE_OUTPUT));

        $retrievedGenome = $agent->getGenomeArray();
        $this->assertCount(2, $retrievedGenome);
    }
}
