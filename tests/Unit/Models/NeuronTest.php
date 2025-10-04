<?php

namespace Rotifer\Tests\Unit\Models;

use PHPUnit\Framework\TestCase;
use Rotifer\Models\Neuron;
use Rotifer\Activations\Activation;

class NeuronTest extends TestCase
{
    public function testNeuronCreation(): void
    {
        $neuron = new Neuron();
        $neuron->setType(Neuron::TYPE_HIDDEN)->setIndex(5);

        $this->assertEquals(Neuron::TYPE_HIDDEN, $neuron->getType());
        $this->assertEquals(5, $neuron->getIndex());
        $this->assertEquals(0, $neuron->getValue());
    }

    public function testNeuronTypes(): void
    {
        $this->assertEquals(0, Neuron::TYPE_INPUT);
        $this->assertEquals(1, Neuron::TYPE_HIDDEN);
        $this->assertEquals(2, Neuron::TYPE_OUTPUT);
    }

    public function testSetAndGetValue(): void
    {
        $neuron = new Neuron();
        $neuron->setValue(3.14);

        $this->assertEquals(3.14, $neuron->getValue());
    }

    public function testActivationFunction(): void
    {
        $neuron = new Neuron();
        $neuron->setValue(0);
        $neuron->applyActivation([Activation::class, 'sigmoid']);

        // sigmoid(0) should be 0.5
        $this->assertEqualsWithDelta(0.5, $neuron->getValue(), 0.001);
    }

    public function testActivationRelu(): void
    {
        $neuron = new Neuron();

        // Test positive value
        $neuron->setValue(5.0);
        $neuron->applyActivation([Activation::class, 'relu']);
        $this->assertEquals(5.0, $neuron->getValue());

        // Test negative value
        $neuron->setValue(-3.0);
        $neuron->applyActivation([Activation::class, 'relu']);
        $this->assertEquals(0.0, $neuron->getValue());
    }

    public function testInConnections(): void
    {
        $neuron = new Neuron();
        $neuron->setInConnection(Neuron::TYPE_INPUT, 3, 2.5);
        $neuron->setInConnection(Neuron::TYPE_HIDDEN, 7, -1.2);

        $connections = $neuron->getInConnections();

        $this->assertArrayHasKey(Neuron::TYPE_INPUT, $connections);
        $this->assertArrayHasKey(Neuron::TYPE_HIDDEN, $connections);
        $this->assertEquals(2.5, $connections[Neuron::TYPE_INPUT][3]);
        $this->assertEquals(-1.2, $connections[Neuron::TYPE_HIDDEN][7]);
    }

    public function testOutConnections(): void
    {
        $neuron = new Neuron();
        $neuron->setOutConnection(Neuron::TYPE_OUTPUT, 2, 0.8);
        $neuron->setOutConnection(Neuron::TYPE_HIDDEN, 5, 1.5);

        $connections = $neuron->getOutConnections();

        $this->assertArrayHasKey(Neuron::TYPE_OUTPUT, $connections);
        $this->assertArrayHasKey(Neuron::TYPE_HIDDEN, $connections);
        $this->assertEquals(0.8, $connections[Neuron::TYPE_OUTPUT][2]);
        $this->assertEquals(1.5, $connections[Neuron::TYPE_HIDDEN][5]);
    }

    public function testDeleteInConnection(): void
    {
        $neuron = new Neuron();
        $neuron->setInConnection(Neuron::TYPE_INPUT, 3, 2.5);
        $neuron->setInConnection(Neuron::TYPE_HIDDEN, 7, -1.2);

        $this->assertTrue($neuron->deleteInConnection(Neuron::TYPE_INPUT, 3));

        $connections = $neuron->getInConnections();
        $this->assertArrayNotHasKey(3, $connections[Neuron::TYPE_INPUT] ?? []);
        $this->assertEquals(-1.2, $connections[Neuron::TYPE_HIDDEN][7]);
    }

    public function testDeleteOutConnection(): void
    {
        $neuron = new Neuron();
        $neuron->setOutConnection(Neuron::TYPE_OUTPUT, 2, 0.8);
        $neuron->setOutConnection(Neuron::TYPE_HIDDEN, 5, 1.5);

        $this->assertTrue($neuron->deleteOutConnection(Neuron::TYPE_OUTPUT, 2));

        $connections = $neuron->getOutConnections();
        $this->assertArrayNotHasKey(2, $connections[Neuron::TYPE_OUTPUT] ?? []);
        $this->assertEquals(1.5, $connections[Neuron::TYPE_HIDDEN][5]);
    }

    public function testDeleteBothConnections(): void
    {
        $neuron = new Neuron();
        $neuron->setInConnection(Neuron::TYPE_INPUT, 1, 1.0);
        $neuron->setOutConnection(Neuron::TYPE_OUTPUT, 1, 1.0);

        $this->assertTrue($neuron->deleteConnection(Neuron::TYPE_INPUT, 1));

        $inConnections = $neuron->getInConnections();
        $outConnections = $neuron->getOutConnections();

        $this->assertArrayNotHasKey(1, $inConnections[Neuron::TYPE_INPUT] ?? []);
        $this->assertEquals(1.0, $outConnections[Neuron::TYPE_OUTPUT][1]);
    }

    public function testDeleteAllConnections(): void
    {
        $neuron = new Neuron();
        $neuron->setInConnection(Neuron::TYPE_INPUT, 1, 1.0);
        $neuron->setOutConnection(Neuron::TYPE_OUTPUT, 1, 1.0);

        $this->assertTrue($neuron->deleteConnections());

        $this->assertEmpty($neuron->getInConnections());
        $this->assertEmpty($neuron->getOutConnections());
    }

    public function testMultipleConnectionsToSameNeuron(): void
    {
        $neuron = new Neuron();
        $neuron->setInConnection(Neuron::TYPE_INPUT, 1, 1.0);
        $neuron->setInConnection(Neuron::TYPE_INPUT, 1, 2.0); // Overwrite

        $connections = $neuron->getInConnections();
        $this->assertEquals(2.0, $connections[Neuron::TYPE_INPUT][1]);
    }
}
