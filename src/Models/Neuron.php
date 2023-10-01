<?php

namespace Rotifer\Models;

use Rotifer\Activations\Activation;

class Neuron
{
    const TYPE_INPUT = 0;
    const TYPE_HIDDEN = 1;
    const TYPE_OUTPUT = 2;

    private int $index;
    private int $type;
    private float $value = 0;

    /**
     * The connections into this neuron
     * @var array [INPUT => [400 => 2.356, 320 => 1.55], NEURON => [506 => 0.056, 601 => 1.2]]
     */
    private array $inConnections = [];

    /**
     * The connections out from this neuron
     * @var array [INPUT => [400 => 2.356, 320 => 1.55], NEURON => [506 => 0.056, 601 => 1.2]]
     */
    private array $outConnections = [];

    public function setType(int $type): static
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

    public function setIndex(int $index): static
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

    public function setInConnection($type, $index, $weight): static
    {
        $this->inConnections[$type][$index] = $weight;

        return $this;
    }

    public function getOutConnections(): array
    {
        return $this->outConnections;
    }

    public function setOutConnection($type, $index, $weight): static
    {
        $this->outConnections[$type][$index] = $weight;

        return $this;
    }

    public function deleteInConnection($type, $index): bool
    {
        unset($this->inConnections[$type][$index]);
        return true;
    }

    public function deleteOutConnection($type, $index): bool
    {
        unset($this->outConnections[$type][$index]);
        return true;
    }

    /**
     * Deletes both in-connection and out-connection
     * @param $type
     * @param $index
     * @return bool
     */
    public function deleteConnection($type, $index): bool
    {
        unset($this->inConnections[$type][$index]);
        unset($this->outConnections[$type][$index]);
        return true;
    }

    public function deleteConnections(): bool
    {
        $this->inConnections = [];
        $this->outConnections = [];

        return true;
    }
}