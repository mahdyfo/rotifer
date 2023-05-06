<?php

namespace GeneticAutoml\Models;

use Closure;
use Exception;
use GeneticAutoml\Encoders\BinaryEncoder;
use GeneticAutoml\Encoders\Encoder;
use GeneticAutoml\Helpers\ReproductionHelper;
use GeneticAutoml\Helpers\WeightHelper;

class Agent
{
    /**
     * Used as a memory database
     * @var array
     */
    private array $neurons = [];
    private $activation = null;

    /**
     * @param int $type The type of the neuron like Neuron::TYPE_INPUT
     * @param int $index
     * @return Neuron
     * @throws Exception
     */
    public function findOrCreateNeuron(int $type, int $index): Neuron
    {
        if ($index < 0 || $index > 65535) {
            throw new Exception('Index ' . $index . ' is out of allowed range of 65535. Current value: ' . $index);
        }

        if (isset($this->neurons[$type][$index])) {
            return $this->neurons[$type][$index];
        }

        return $this->neurons[$type][$index] = (new Neuron())->setType($type)->setIndex($index);
    }

    public function createNeuron(int $type, $count = 1): self
    {
        for ($i = 0; $i < $count; $i++) {
            if (empty($this->neurons[$type])) {
                $index = 0;
            } else {
                $index = max(array_keys($this->neurons[$type])) + 1;
            }
            $this->neurons[$type][$index] = (new Neuron())->setType($type)->setIndex($index);
        }

        return $this;
    }

    /**
     * Connect all inputs to all outputs with random weights
     * @return Agent
     * @throws Exception
     */
    public function initRandomConnections(): self
    {
        /** @var Neuron $inputNeuron */
        /** @var Neuron $outputNeuron */
        foreach ($this->getNeuronsByType(Neuron::TYPE_INPUT) as $inputNeuron) {
            foreach ($this->getNeuronsByType(Neuron::TYPE_OUTPUT) as $outputNeuron) {
                $inputNeuron->connectTo($outputNeuron, WeightHelper::generateRandomWeight());
            }
        }

        return $this;
    }

    public function getNeuronsByType(int $type): array
    {
        if (isset($this->neurons[$type])) {
            ksort($this->neurons[$type]);
            return $this->neurons[$type];
        }

        return [];
    }

    public function getAllNeurons(): array
    {
        return $this->neurons;
    }

    public function setActivation($activation): self
    {
        $this->activation = $activation;

        return $this;
    }

    /**
     * @param Encoder|null $encoder Default BinaryEncoder
     * @param Closure|null $iterationCallback The function to execute on each element
     * @return array
     */
    public function getGenomeArray(Encoder $encoder = null, Closure $iterationCallback = null): array
    {
        if (is_null($encoder)) {
            $encoder = new BinaryEncoder();
        }

        $neurons = $this->getNeuronsByType(Neuron::TYPE_HIDDEN);
        $neurons = array_merge($neurons, $this->getNeuronsByType(Neuron::TYPE_OUTPUT));
        $genomes = [];
        /** @var Neuron $neuron */
        foreach ($neurons as $neuron) {
            foreach ($neuron->getInConnections() as $fromType => $indexConnections) {
                foreach ($indexConnections as $fromIndex => $weight) {
                    $genome = $encoder->encodeConnection(
                        $fromType,
                        $fromIndex,
                        $neuron->getType(),
                        $neuron->getIndex(),
                        $weight
                    );
                    $genomes[] = !is_null($iterationCallback) ? $iterationCallback($genome) : $genome;
                }
            }
        }
        return $genomes;
    }

    public function getGenomeString(Encoder $encoder = null, $separator = ';', $iterationCallback = null): string
    {
        return implode($separator, $this->getGenomeArray($encoder, $iterationCallback));
    }

    /**
     * @param array $inputs
     * @return void
     * @throws Exception
     */
    public function step(array $inputs): void
    {
        // Set inputs
        $inputNeurons = $this->getNeuronsByType(Neuron::TYPE_INPUT);
        foreach ($inputs as $key => $inputValue) {
            /** @var Neuron[] $inputNeurons */
            $inputNeurons[$key]->setValue($inputValue);
        }

        // Calculate neurons
        $neuronGroups = [
            $this->getNeuronsByType(Neuron::TYPE_HIDDEN),
            $this->getNeuronsByType(Neuron::TYPE_OUTPUT)
        ];
        foreach ($neuronGroups as $neuronGroup) {
            // Foreach all hidden neurons
            foreach ($neuronGroup as $neuron) {
                /** @var Neuron $neuron */
                $newValue = 0;
                // Foreach inward connections [INPUT => [435 => 2.6, 266 => 1.4], NEURON => [...]]
                foreach ($neuron->getInConnections() as $type => $neuronsByIndex) {
                    foreach ($neuronsByIndex as $index => $weight) {
                        // Neuron value += inward neuron value * weight
                        $newValue += $this->findOrCreateNeuron($type, $index)->getValue() * $weight;
                    }
                }
                $neuron->setValue($newValue)->applyActivation($this->activation);
            }
        }
    }

    public function reproduce(Agent $agent)
    {
        $child = ReproductionHelper::crossover($this, $agent);

    }
}