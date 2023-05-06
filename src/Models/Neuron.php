<?php

namespace GeneticAutoml\Models;

use Exception;
use GeneticAutoml\Activations\Activation;
use GeneticAutoml\Helpers\WeightHelper;

class Neuron
{
    const TYPE_INPUT = 0;
    const TYPE_HIDDEN = 1;
    const TYPE_OUTPUT = 2;

    private int $index;
    private int $type;
    private float $value = 0;

    // [INPUT => [400 => 2.356, 320 => 1.55], NEURON => [506 => 0.056, 601 => 1.2]]
    private array $inConnections = [];
    private array $outConnections = [];



    public function connectTo(Neuron $neuron, $weight): self
    {
        if ($weight > WeightHelper::MAX_WEIGHT || $weight < -WeightHelper::MAX_WEIGHT) {
            throw new Exception('Weight of connection from ' . $this->getType() . ':' . $this->getIndex()
                . ' to ' . $this->getType() . ':' . $this->getIndex()
                . ' is out of allowed range of +-' . WeightHelper::MAX_WEIGHT . '. Current value: ' . $weight);
        }

        if ($this->getType() == Neuron::TYPE_INPUT && $neuron->getType() == Neuron::TYPE_INPUT) {
            throw new Exception('Cannot connect input to input');
        }
        if ($this->getType() == Neuron::TYPE_OUTPUT && $neuron->getType() == Neuron::TYPE_OUTPUT) {
            throw new Exception('Cannot connect output to output');
        }
        if ($this->getType() == Neuron::TYPE_HIDDEN && $neuron->getType() == Neuron::TYPE_INPUT) {
            throw new Exception('Cannot connect hidden to input');
        }

        $this->outConnections[$neuron->getType()][$neuron->getIndex()] = $weight;
        $neuron->inConnections[$this->getType()][$this->getIndex()] = $weight;

        return $this;
    }

    public function setType(int $type): self
    {
        $this->type = $type;

        return $this;
    }

    public function getType(): int
    {
        return $this->type;
    }

    public function setValue(float $value): self
    {
        $this->value = $value;

        return $this;
    }

    public function getValue(): float
    {
        return $this->value;
    }

    public function applyActivation($activation = null): void
    {
        if (empty($activation)) {
            $activation = [Activation::class, 'sigmoid'];
        }
        $this->setValue($activation($this->getValue()));
    }

    public function setIndex(int $index): self
    {
        $this->index = $index;

        return $this;
    }

    public function getIndex(): int
    {
        return $this->index;
    }

    public function getInConnections(): array
    {
        return $this->inConnections;
    }
}