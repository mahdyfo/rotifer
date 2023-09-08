<?php

namespace Rotifer\Models;

use Exception;
use Rotifer\Helpers\WeightHelper;

/**
 * StaticAgent is an agent that doesn't have dynamic hidden neurons. It has traditional unchanged neurons and layers
 */
class StaticAgent extends Agent
{
    /**
     * @var array $layers For example 3 layers with 5, 4, 5 neurons in each: [5, 4, 5]
     */
    private array $layers = [];
    /**
     * Connect all inputs to all outputs with random weights
     * @return StaticAgent
     * @throws Exception
     */
    public function initRandomConnections(): static
    {
        $inputNeurons = $this->getNeuronsByType(Neuron::TYPE_INPUT);
        $hiddenNeurons = $this->getNeuronsByType(Neuron::TYPE_HIDDEN);
        $outputNeurons = $this->getNeuronsByType(Neuron::TYPE_OUTPUT);

        $layerNeurons = [];
        $i = 0;
        foreach ($this->getLayers() as $layerNeuronsCount) {
            $layerNeurons[] = array_slice($hiddenNeurons, $i, $layerNeuronsCount, true);
            $i += $layerNeuronsCount;
        }

        // Input to first hidden layer
        foreach ($inputNeurons as $inputNeuron) {
            foreach ($layerNeurons[0] as $hiddenNeuron) {
                $this->connectNeurons($inputNeuron, $hiddenNeuron, WeightHelper::generateRandomWeight());
            }
        }

        // Hidden layer to next hidden layer
        if (count($layerNeurons) > 1) {
            for ($i = 0; $i < count($layerNeurons) - 1; $i++) {
                foreach ($layerNeurons[$i] as $hiddenNeuron1) {
                    foreach ($layerNeurons[$i + 1] as $hiddenNeuron2) {
                        $this->connectNeurons($hiddenNeuron1, $hiddenNeuron2, WeightHelper::generateRandomWeight());
                    }
                }
            }
        }

        // Last hidden layer to outputs
        foreach ($outputNeurons as $outputNeuron) {
            foreach ($layerNeurons[array_key_last($layerNeurons)] as $hiddenNeuron) {
                $this->connectNeurons($hiddenNeuron, $outputNeuron, WeightHelper::generateRandomWeight());
            }
        }

        return $this;
    }

    /**
     * @param array $layers For example 3 layers with 5, 4, 5 neurons in each: [5, 4, 5]
     * @return $this
     */
    public function createHiddenLayerNeurons(array $layers = []): static
    {
        $this->layers = $layers;

        // If count is array, it indicates layers neurons, so the total count will be sum of the array
        $count = array_sum($layers);

        return $this->createNeuron(Neuron::TYPE_HIDDEN, $count, false);
    }

    public function setLayers(array $layers)
    {
        $this->layers = $layers;
    }

    public function getLayers(): array
    {
        return $this->layers;
    }

    /**
     * @param int $layerIndex Which hidden layer should it return the neurons from. (starts from 0)
     * @return Neuron[]
     */
    public function getNeuronsByLayer(int $layerIndex): array
    {
        if (isset($this->layers[$layerIndex])) {
            $hiddenNeurons = $this->getNeuronsByType(Neuron::TYPE_HIDDEN, false);
            // Calculate start and end index of layer neurons
            $startIndex = 0;
            // Example layers: [4, 5, 3], i = 1 (second layer)
            for ($i = 0; $i < $layerIndex; $i++) {
                // start index = 4 for second layer
                $startIndex += $this->layers[$i];
            }
            return array_slice($hiddenNeurons, $startIndex, $this->layers[$layerIndex], true);
        }

        return [];
    }
}
