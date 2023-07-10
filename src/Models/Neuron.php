<?php

namespace GeneticAutoml\Models;

use GeneticAutoml\Activations\Activation;

class Neuron
{
    const TYPE_INPUT = 0;
    const TYPE_HIDDEN = 1;
    const TYPE_OUTPUT = 2;

    private int $index;
    private int $type;
    private float $previousValue = 0;
    private float $value = 0;

    /**
     * The connections into this neuron
     * @var array [INPUT => [400 => 2.356, 320 => 1.55], NEURON => [506 => 0.056, 601 => 1.2]]
     */
    private array $inConnections = [];
    private array $outConnections = [];

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
        $this->setPreviousValue($this->value);

        $this->value = $value;

        return $this;
    }

    public function getValue(): float
    {
        return $this->value;
    }

    public function setPreviousValue(float $value): self
    {
        $this->previousValue = $value;

        return $this;
    }

    public function getPreviousValue(): float
    {
        return $this->previousValue;
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

    public function setInConnection($type, $index, $weight): self
    {
        $this->inConnections[$type][$index] = $weight;

        return $this;
    }

    public function getOutConnections(): array
    {
        return $this->outConnections;
    }

    public function setOutConnection($type, $index, $weight): self
    {
        $this->outConnections[$type][$index] = $weight;

        return $this;
    }

    public function deleteConnections(): array
    {
        return $this->inConnections = [];
    }
}