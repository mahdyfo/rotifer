<?php

namespace Rotifer\Tests\Integration;

use PHPUnit\Framework\TestCase;
use Rotifer\Models\Agent;
use Rotifer\Models\Neuron;
use Rotifer\GeneEncoders\HexEncoder;
use Rotifer\GeneEncoders\BinaryEncoder;
use Rotifer\GeneEncoders\HumanEncoder;
use Rotifer\GeneEncoders\JsonEncoder;
use Rotifer\Activations\Activation;

if (!defined('ACTIVATION')) {
    define('ACTIVATION', [Activation::class, 'sigmoid']);
}
if (!defined('CALCULATE_STEP_TIME')) {
    define('CALCULATE_STEP_TIME', false);
}

class GenomeEncodingTest extends TestCase
{
    private function createTestAgent(): Agent
    {
        $agent = new Agent();
        $agent->createNeuron(Neuron::TYPE_INPUT, 2);
        $agent->createNeuron(Neuron::TYPE_HIDDEN, 2);
        $agent->createNeuron(Neuron::TYPE_OUTPUT, 1);

        $genome = [
            ['from_type' => Neuron::TYPE_INPUT, 'from_index' => 0, 'to_type' => Neuron::TYPE_HIDDEN, 'to_index' => 0, 'weight' => 1.5],
            ['from_type' => Neuron::TYPE_INPUT, 'from_index' => 1, 'to_type' => Neuron::TYPE_HIDDEN, 'to_index' => 1, 'weight' => -2.3],
            ['from_type' => Neuron::TYPE_HIDDEN, 'from_index' => 0, 'to_type' => Neuron::TYPE_OUTPUT, 'to_index' => 0, 'weight' => 0.8],
            ['from_type' => Neuron::TYPE_HIDDEN, 'from_index' => 1, 'to_type' => Neuron::TYPE_OUTPUT, 'to_index' => 0, 'weight' => 0.5],
        ];

        $agent->setGenome($genome);

        return $agent;
    }

    public function testHexEncoderEncodeAndDecode(): void
    {
        $agent = $this->createTestAgent();
        $encoder = HexEncoder::getInstance();

        $originalGenome = $agent->getGenomeArray();
        $encodedString = $agent->getGenomeString($encoder);

        $this->assertIsString($encodedString);
        $this->assertNotEmpty($encodedString);

        // Decode back
        $decodedAgent = Agent::createFromGenome($encodedString, false, $encoder);
        $decodedGenome = $decodedAgent->getGenomeArray();

        $this->assertCount(count($originalGenome), $decodedGenome);

        // Check structure is preserved
        foreach ($decodedGenome as $i => $gene) {
            $this->assertEquals($originalGenome[$i]['from_type'], $gene['from_type']);
            $this->assertEquals($originalGenome[$i]['to_type'], $gene['to_type']);
            $this->assertEqualsWithDelta($originalGenome[$i]['weight'], $gene['weight'], 0.01);
        }
    }

    public function testBinaryEncoderEncodeAndDecode(): void
    {
        $agent = $this->createTestAgent();
        $encoder = BinaryEncoder::getInstance();

        $originalGenome = $agent->getGenomeArray();
        $encodedString = $agent->getGenomeString($encoder);

        $this->assertIsString($encodedString);
        $this->assertNotEmpty($encodedString);

        // Decode back
        $decodedAgent = Agent::createFromGenome($encodedString, false, $encoder);
        $decodedGenome = $decodedAgent->getGenomeArray();

        $this->assertCount(count($originalGenome), $decodedGenome);
    }

    public function testHumanEncoderEncodeAndDecode(): void
    {
        $agent = $this->createTestAgent();
        $encoder = HumanEncoder::getInstance();

        $originalGenome = $agent->getGenomeArray();
        $encodedString = $agent->getGenomeString($encoder);

        $this->assertIsString($encodedString);
        $this->assertNotEmpty($encodedString);

        // Human encoder should be readable (lowercase)
        $this->assertStringContainsString('input', $encodedString);
        $this->assertStringContainsString('weight', $encodedString);

        // Note: HumanEncoder is display-only, not meant for decoding
    }

    public function testJsonEncoderEncodeAndDecode(): void
    {
        $agent = $this->createTestAgent();
        $encoder = JsonEncoder::getInstance();

        $originalGenome = $agent->getGenomeArray();
        $encodedArray = $agent->getGenomeArray($encoder);

        $this->assertIsArray($encodedArray);
        $this->assertNotEmpty($encodedArray);

        // Each gene should be valid JSON when encoded individually
        foreach ($encodedArray as $encodedGene) {
            $this->assertJson($encodedGene);
        }

        // JsonEncoder works per-gene, not for full genome strings
        // Test individual gene encoding/decoding
        foreach ($originalGenome as $i => $gene) {
            $encoded = $encoder->encodeConnection(
                $gene['from_type'],
                $gene['from_index'],
                $gene['to_type'],
                $gene['to_index'],
                $gene['weight']
            );
            $this->assertJson($encoded);

            $decoded = $encoder->decodeConnection($encoded);
            $this->assertEquals($gene['from_type'], $decoded['from_type']);
            $this->assertEquals($gene['to_type'], $decoded['to_type']);
        }
    }

    public function testEncoderSingleton(): void
    {
        $encoder1 = HexEncoder::getInstance();
        $encoder2 = HexEncoder::getInstance();

        $this->assertSame($encoder1, $encoder2);
    }

    public function testEncodedGenomePreservesNeuronCounts(): void
    {
        $agent = $this->createTestAgent();
        $encoder = HexEncoder::getInstance();

        $originalInputCount = count($agent->getNeuronsByType(Neuron::TYPE_INPUT));
        $originalHiddenCount = count($agent->getNeuronsByType(Neuron::TYPE_HIDDEN));
        $originalOutputCount = count($agent->getNeuronsByType(Neuron::TYPE_OUTPUT));

        $encodedString = $agent->getGenomeString($encoder);
        $decodedAgent = Agent::createFromGenome($encodedString, false, $encoder);

        // Genome encoding preserves the neurons referenced in connections
        // Input and output counts should match
        $this->assertCount($originalInputCount, $decodedAgent->getNeuronsByType(Neuron::TYPE_INPUT));
        $this->assertCount($originalOutputCount, $decodedAgent->getNeuronsByType(Neuron::TYPE_OUTPUT));
        // Hidden neurons: only those with connections are preserved
        $this->assertGreaterThan(0, count($decodedAgent->getNeuronsByType(Neuron::TYPE_HIDDEN)));
    }

    public function testEncodedGenomeProducesIdenticalOutput(): void
    {
        $agent = $this->createTestAgent();
        $encoder = HexEncoder::getInstance();

        $inputs = [1.0, 0.5];

        // Get outputs from original agent
        $agent->step($inputs);
        $originalOutputs = $agent->getOutputValues();

        // Encode and decode
        $encodedString = $agent->getGenomeString($encoder);
        $decodedAgent = Agent::createFromGenome($encodedString, false, $encoder);

        // Get outputs from decoded agent
        $decodedAgent->step($inputs);
        $decodedOutputs = $decodedAgent->getOutputValues();

        // Outputs should be very similar (within small delta due to encoding precision)
        $this->assertCount(count($originalOutputs), $decodedOutputs);
        for ($i = 0; $i < count($originalOutputs); $i++) {
            $this->assertEqualsWithDelta($originalOutputs[$i], $decodedOutputs[$i], 0.01);
        }
    }

    public function testMultipleEncodersProduceSameBehavior(): void
    {
        $agent = $this->createTestAgent();

        $inputs = [0.7, 0.3];
        $agent->step($inputs);
        $originalOutputs = $agent->getOutputValues();

        // Only test encoders that support bidirectional encoding/decoding
        $encoders = [
            HexEncoder::getInstance(),
            BinaryEncoder::getInstance(),
        ];

        foreach ($encoders as $encoder) {
            $encodedString = $agent->getGenomeString($encoder);
            $decodedAgent = Agent::createFromGenome($encodedString, false, $encoder);

            $decodedAgent->step($inputs);
            $decodedOutputs = $decodedAgent->getOutputValues();

            // All encoders should produce similar outputs
            $this->assertEqualsWithDelta($originalOutputs[0], $decodedOutputs[0], 0.1);
        }
    }

    public function testEmptyGenomeEncoding(): void
    {
        $agent = new Agent();
        $agent->createNeuron(Neuron::TYPE_INPUT, 1);
        $agent->createNeuron(Neuron::TYPE_OUTPUT, 1);
        // No connections

        $encoder = HexEncoder::getInstance();
        $encodedString = $agent->getGenomeString($encoder);

        // Empty genome should produce empty or minimal string
        $this->assertIsString($encodedString);
    }

    public function testLargeGenomeEncoding(): void
    {
        $agent = new Agent();
        $agent->createNeuron(Neuron::TYPE_INPUT, 10);
        $agent->createNeuron(Neuron::TYPE_HIDDEN, 20);
        $agent->createNeuron(Neuron::TYPE_OUTPUT, 5);
        $agent->initRandomConnections();

        $encoder = HexEncoder::getInstance();

        $originalGenome = $agent->getGenomeArray();
        $encodedString = $agent->getGenomeString($encoder);
        $decodedAgent = Agent::createFromGenome($encodedString, false, $encoder);
        $decodedGenome = $decodedAgent->getGenomeArray();

        $this->assertCount(count($originalGenome), $decodedGenome);
    }

    public function testEncodingWithMemory(): void
    {
        $agent = $this->createTestAgent();
        $agent->setHasMemory(true);

        $encoder = HexEncoder::getInstance();
        $encodedString = $agent->getGenomeString($encoder);

        $decodedAgent = Agent::createFromGenome($encodedString, true, $encoder);

        $this->assertTrue($decodedAgent->hasMemory());
    }

    public function testGenomeArrayFormat(): void
    {
        $agent = $this->createTestAgent();
        $genome = $agent->getGenomeArray();

        foreach ($genome as $gene) {
            $this->assertArrayHasKey('from_type', $gene);
            $this->assertArrayHasKey('from_index', $gene);
            $this->assertArrayHasKey('to_type', $gene);
            $this->assertArrayHasKey('to_index', $gene);
            $this->assertArrayHasKey('weight', $gene);

            $this->assertIsInt($gene['from_type']);
            $this->assertIsInt($gene['from_index']);
            $this->assertIsInt($gene['to_type']);
            $this->assertIsInt($gene['to_index']);
            $this->assertIsFloat($gene['weight']);
        }
    }
}
