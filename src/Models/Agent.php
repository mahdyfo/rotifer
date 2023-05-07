<?php

namespace GeneticAutoml\Models;

use Exception;
use GeneticAutoml\Encoders\Encoder;
use GeneticAutoml\Helpers\ReproductionHelper;
use GeneticAutoml\Helpers\WeightHelper;

class Agent
{
    /**
     * @var Neuron[][]
     */
    private array $neurons = [];

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
     * @param array|string $genome raw array or string encoded genome
     * @param Encoder|null $decoder
     * @param string $separator If you provide string, you should specify the separator between genes
     * @return Agent
     * @throws Exception
     */
    public static function createFromGenome(array|string $genome, Encoder $decoder = null, string $separator = ';'): self
    {
        return (new self())->setGenome($genome, $decoder, $separator);
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
                $this->connectNeurons($inputNeuron, $outputNeuron, WeightHelper::generateRandomWeight());
            }
        }

        return $this;
    }

    public function connectNeurons(Neuron $neuron1, Neuron $neuron2, $weight): self
    {
        if ($weight > WeightHelper::MAX_WEIGHT || $weight < -WeightHelper::MAX_WEIGHT) {
            throw new Exception('Weight of connection from ' . $neuron1->getType() . ':' . $neuron1->getIndex()
                . ' to ' . $neuron1->getType() . ':' . $neuron1->getIndex()
                . ' is out of allowed range of +-' . WeightHelper::MAX_WEIGHT . '. Current value: ' . $weight);
        }

        if ($neuron1->getType() == Neuron::TYPE_INPUT && $neuron2->getType() == Neuron::TYPE_INPUT) {
            throw new Exception('Cannot connect input to input');
        }
        if ($neuron1->getType() == Neuron::TYPE_OUTPUT && $neuron2->getType() == Neuron::TYPE_OUTPUT) {
            throw new Exception('Cannot connect output to output');
        }
        if ($neuron1->getType() == Neuron::TYPE_HIDDEN && $neuron2->getType() == Neuron::TYPE_INPUT) {
            throw new Exception('Cannot connect hidden to input');
        }

        $neuron2->setInConnection($neuron1->getType(), $neuron1->getIndex(), $weight);
        $neuron1->setOutConnection($neuron2->getType(), $neuron2->getIndex(), $weight);

        return $this;
    }

    /**
     * @param array|string $genome raw array or string encoded genome
     * @param Encoder|null $decoder
     * @param string $separator If you provide string, you should specify the separator between genes
     * @return Agent
     * @throws Exception
     */
    public function setGenome(array|string $genome, Encoder $decoder = null, string $separator = ';'): self
    {
        // Separate genes if genome is provided as string
        if (!is_array($genome)) {
            $genome = array_filter(explode($separator, $genome));
        }

        // Decode the genes if a decoder is provided
        if (!is_null($decoder)) {
            foreach ($genome as $key => $gene) {
                $genome[$key] = $decoder->decodeConnection($gene);
            }
        }

        // Delete all previous connections but do not delete neurons because they have previous calculated values in them
        $this->deleteAllConnections();

        // Create new connections from genome
        foreach ($genome as $gene) {
            $this->connectNeurons(
                $this->findOrCreateNeuron($gene['from_type'], $gene['from_index']),
                $this->findOrCreateNeuron($gene['to_type'], $gene['to_index']),
                $gene['weight']
            );
        }

        $this->deleteRedundantNeurons();

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

    /**
     * @param Encoder|null $encoder Default is array genome
     * @param mixed $iterationCallback The function to execute on each element
     * @return array
     */
    public function getGenomeArray(Encoder $encoder = null, $iterationCallback = null): array
    {
        $neuronGroups = [
            $this->getNeuronsByType(Neuron::TYPE_HIDDEN),
            $this->getNeuronsByType(Neuron::TYPE_OUTPUT)
        ];
        $genomes = [];
        /** @var Neuron $neuron */
        foreach ($neuronGroups as $neuronGroup) {
            foreach ($neuronGroup as $neuron) {
                foreach ($neuron->getInConnections() as $fromType => $indexConnections) {
                    foreach ($indexConnections as $fromIndex => $weight) {
                        $genome = [
                            'from_type' => $fromType,
                            'from_index' => $fromIndex,
                            'to_type' => $neuron->getType(),
                            'to_index' => $neuron->getIndex(),
                            'weight' => $weight
                        ];
                        if (!is_null($encoder)) {
                            $genome = $encoder->encodeConnection(...$genome);
                        }
                        $genomes[] = !is_null($iterationCallback) ? $iterationCallback($genome) : $genome;
                    }
                }
            }
        }
        return $genomes;
    }

    /**
     * Get string formatted genome
     * @param Encoder $encoder You must specify encoder because you want a string
     * @param string $separator
     * @param null $iterationCallback
     * @return string
     */
    public function getGenomeString(Encoder $encoder, string $separator = ';', $iterationCallback = null): string
    {
        return implode($separator, $this->getGenomeArray($encoder, $iterationCallback));
    }

    public function deleteAllConnections(): void
    {
        foreach ($this->neurons as $type => $neuronsByIndex) {
            foreach ($neuronsByIndex as $index => $neuron) {
                $neuron->deleteConnections();
            }
        }
    }

    /**
     * Delete stray hidden neurons without any out-connections, or just 1 out-connection to themselves
     * @return void
     */
    public function deleteRedundantNeurons(): void
    {
        $neuronsByIndex = $this->getNeuronsByType(Neuron::TYPE_HIDDEN);
        foreach ($neuronsByIndex as $index => $neuron) {
            // Delete stray neurons without any out-connections
            if (count($neuron->getOutConnections()) == 0) {
                unset($this->neurons[Neuron::TYPE_HIDDEN][$index]);
            }

            // Delete neurons with 1 only out-connection just to themselves
            if (
                count($neuron->getOutConnections()) == 1 &&
                isset($neuron->getOutConnections()[Neuron::TYPE_HIDDEN][$index])
            ) {
                unset($this->neurons[Neuron::TYPE_HIDDEN][$index]);
            }
        }

        if (empty($this->neurons[Neuron::TYPE_HIDDEN])) {
            unset($this->neurons[Neuron::TYPE_HIDDEN]);
        }
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
                $neuron->setValue($newValue)->applyActivation();
            }
        }

        //TODO: calculate fitness score based on fitness callback function
    }
}