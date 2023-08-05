<?php

namespace GeneticAutoml\Models;

use Exception;
use GeneticAutoml\Encoders\Encoder;
use GeneticAutoml\Helpers\WeightHelper;

class Agent
{
    /**
     * neurons[type][index]
     * @var Neuron[][]
     */
    private array $neurons = [];
    private float $fitness = 0;
    private int $step = 0;
    private float $stepTime = 0;
    private bool $hasMemory = false;
    private array $additional = [];

    public function findNeuron(int $type, int $index): ?Neuron
    {
        if ($index < 0 || $index > 65535) {
            throw new Exception('Index ' . $index . ' is out of allowed range of 65535. Current value: ' . $index);
        }

        // Find
        if (isset($this->neurons[$type][$index])) {
            return $this->neurons[$type][$index];
        }

        return null;
    }

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

        // Find
        $neuron = $this->findNeuron($type, $index);
        if (!is_null($neuron)) {
            return $this->neurons[$type][$index];
        }

        // Create
        return $this->neurons[$type][$index] = (new Neuron())->setType($type)->setIndex($index);
    }

    public function createNeuron(int $type, int $count = 1, bool $connectToAll = false): self
    {
        for ($i = 0; $i < $count; $i++) {
            if (empty($this->neurons[$type])) {
                $index = 0;
            } else {
                $index = max(array_keys($this->neurons[$type])) + 1;
            }
            $this->neurons[$type][$index] = (new Neuron())->setType($type)->setIndex($index);

            if ($connectToAll && $type == Neuron::TYPE_HIDDEN) {
                $this->connectToAll($this->neurons[$type][$index]);
            }
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
     * Connects the input neuron to all other neurons and inputs and outputs with random weight
     * Other neurons will be connected to the new hidden one. Because we calculate neurons from index 0 and
     * connecting new neuron to other hidden ones makes no sense. But connecting other ones to new one makes sense.
     * @param Neuron $inputNeuron
     * @return $this
     * @throws Exception
     */
    public function connectToAll(Neuron $inputNeuron): self
    {
	$targetType = $inputNeuron->getType();

        foreach ($this->neurons as $sourceType => $neurons) {
            foreach ($neurons as $index => $neuron) {
	            // Inputs -> target hidden or target output
                if ($sourceType == Neuron::TYPE_INPUT && ($targetType == Neuron::TYPE_HIDDEN || $targetType == Neuron::TYPE_OUTPUT)) {
                    $this->connectNeurons($neuron, $inputNeuron, WeightHelper::generateRandomWeight());
                }
                // Target input or Target hidden -> Outputs
                if ($sourceType == Neuron::TYPE_OUTPUT && ($targetType == Neuron::TYPE_INPUT || $targetType == Neuron::TYPE_HIDDEN)) {
                    $this->connectNeurons($inputNeuron, $neuron, WeightHelper::generateRandomWeight());
                }
		        // Hiddens -> target hidden | Inputs -> target hidden | Target input -> hiddens
                if ($sourceType == Neuron::TYPE_HIDDEN) {
                    // Do not connect to itself if agent has no memory
                    if (!$this->hasMemory() && $neuron->getIndex() == $inputNeuron->getIndex()) {
                        continue;
                    }
                    // Connect other neurons to the new neuron
                    // Target input -> hiddens
                    if ($targetType == Neuron::TYPE_INPUT) {
                        $this->connectNeurons($inputNeuron, $neuron, WeightHelper::generateRandomWeight());
                    }
                    // Inputs or Hiddens -> Target output
                    if ($targetType == Neuron::TYPE_OUTPUT) {
                        $this->connectNeurons($neuron, $inputNeuron, WeightHelper::generateRandomWeight());
                    }
                    // Hiddens -> Target hidden
                    if ($targetType == Neuron::TYPE_HIDDEN) {
                        $this->connectNeurons($neuron, $inputNeuron, WeightHelper::generateRandomWeight());
                    }
                }
            }
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

        $this->deleteRedundantGenes();

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

    public function removeNeuron(int $type, int $index): self
    {
        unset($this->neurons[$type][$index]);

        return $this;
    }

    /**
     * @param Encoder|null $encoder Default is array genome
     * @param mixed $iterationCallback The function to execute on each element
     * @return array [gene, gene]
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
                            $genome = $encoder->encodeConnection(...array_values($genome));
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
    public function deleteRedundantGenes(): void
    {
        // If no input or output connections, delete everything
        if (empty($this->neurons[Neuron::TYPE_INPUT]) || empty($this->neurons[Neuron::TYPE_OUTPUT])) {
            $this->deleteAllConnections();
            return;
        }

        // No need to remove duplicates. They are automatically replaced because of having same array keys
        $neuronsByIndex = $this->getNeuronsByType(Neuron::TYPE_HIDDEN);
        foreach ($neuronsByIndex as $index => $neuron) {
            // If no memory, delete self-connections and all future in-connections (neuron to neuron) starting from beginning. Because they don't have any effect.
            if (!$this->hasMemory()) {
                $inConnections = $neuron->getInConnections()[Neuron::TYPE_HIDDEN] ?? [];
                foreach ($inConnections as $otherIndex => $inConnection) {
                    // Remove self-connection
                    if ($index == $otherIndex) {
                        $this->neurons[Neuron::TYPE_HIDDEN][$index]->deleteConnection(Neuron::TYPE_HIDDEN, $index);
                    }
                    // Remove in-connection from future neurons
                    if ($otherIndex > $index) {
                        $this->neurons[Neuron::TYPE_HIDDEN][$index]->deleteInConnection(Neuron::TYPE_HIDDEN, $otherIndex);
                    }
                }
            }

            // Delete neurons with 1 only out-connection just to themselves
            if (
                count($neuron->getOutConnections()) == 1 &&
                isset($neuron->getOutConnections()[Neuron::TYPE_HIDDEN][$index])
            ) {
                unset($this->neurons[Neuron::TYPE_HIDDEN][$index]);
            }

            // Delete neurons with 1 only in-connection just from themselves
            if (
                count($neuron->getInConnections()) == 1 &&
                isset($neuron->getOutConnections()[Neuron::TYPE_HIDDEN][$index])
            ) {
                unset($this->neurons[Neuron::TYPE_HIDDEN][$index]);
            }

            // Delete stray neurons without any out-connections
            if (count($neuron->getOutConnections()) == 0) {
                unset($this->neurons[Neuron::TYPE_HIDDEN][$index]);
            }

            // Delete stray neurons without any in & out-connections
            if (count($neuron->getInConnections()) == 0 && count($neuron->getOutConnections()) == 0) {
                unset($this->neurons[Neuron::TYPE_HIDDEN][$index]);
            }
        }

        if (empty($this->neurons[Neuron::TYPE_HIDDEN])) {
            unset($this->neurons[Neuron::TYPE_HIDDEN]);
        }
    }

    public function resetValues(): self
    {
        foreach ($this->getNeuronsByType(Neuron::TYPE_HIDDEN) as $neuron) {
            $neuron->setValue(0);
        }

        return $this;
    }

    /**
     * @param array $inputs
     * @return Agent
     * @throws Exception
     */
    public function step(array $inputs): Agent
    {
        if (CALCULATE_STEP_TIME) {
            $time1 = microtime(true);
        }

        /** @var Neuron $neuron */
        // Set inputs
        $inputNeurons = $this->getNeuronsByType(Neuron::TYPE_INPUT);
        foreach ($inputNeurons as $key => $inputNeuron) {
            /** @var Neuron $inputNeuron */
            $inputNeuron->setValue($inputs[$key]);
        }

        if (!$this->hasMemory()) {
            // If the agent doesn't have memory, init hidden neurons with zero
            foreach ($this->getNeuronsByType(Neuron::TYPE_HIDDEN) as $neuron) {
                $neuron->setValue(0);
            }
        }

        // Calculate neurons
        $neuronGroups = [
            $this->getNeuronsByType(Neuron::TYPE_HIDDEN),
            $this->getNeuronsByType(Neuron::TYPE_OUTPUT)
        ];
        foreach ($neuronGroups as $neuronGroup) {
            // Foreach all hidden neurons
            foreach ($neuronGroup as $neuron) {
                if ($this->hasMemory() && $neuron->getType() == Neuron::TYPE_HIDDEN) {
                    // If the agent has memory, use dropout technique (20%) to avoid over-fitting or learning the train-set
                    if (mt_rand(1, 100) <= 20) {
                        // Delete the neuron memory
                        $neuron->setValue(0);
                        // Don't calculate
                        continue;
                    }
                }
                $newValue = 0;
                // Foreach inward connections [INPUT => [435 => 2.6, 266 => 1.4], NEURON => [...]]
                foreach ($neuron->getInConnections() as $type => $neuronsByIndex) {
                    foreach ($neuronsByIndex as $index => $weight) {
                        // Neuron value += inward neuron value * weight
                        $newValue += $this->findOrCreateNeuron($type, $index)->getValue() * $weight;
                    }
                }
                $neuron->setValue($newValue)->applyActivation(ACTIVATION);
            }
        }

        $this->step++;

        if (CALCULATE_STEP_TIME) {
            $this->stepTime = microtime(true) - $time1;
        }

        return $this;
    }

    /**
     * Get the current inputs of the agent
     * @return array
     */
    public function getInputValues(): array
    {
        $inputs = [];
        foreach ($this->getNeuronsByType(Neuron::TYPE_INPUT) as $input) {
            $inputs[] = $input->getValue();
        }
        return $inputs;
    }

    /**
     * Get the current outputs of the agent
     * @return array
     */
    public function getOutputValues(): array
    {
        $outputs = [];
        foreach ($this->getNeuronsByType(Neuron::TYPE_OUTPUT) as $output) {
            $outputs[] = $output->getValue();
        }
        return $outputs;
    }

    /**
     * Use it to know the current step of the agent
     * @return int
     */
    public function getStep(): int
    {
        return $this->step;
    }

    /**
     * The time it took from start to finish of the step (microtime)
     * @return float
     */
    public function getStepTime(): float
    {
        return $this->stepTime;
    }

    /**
     * Get the latest calculated fitness value of the agent
     * @return float
     */
    public function getFitness(): float
    {
        return $this->fitness;
    }

    public function setFitness(float $fitness): self
    {
        $this->fitness = $fitness;

        return $this;
    }

    public function setHasMemory(bool $hasMemory = false): self
    {
        $this->hasMemory = $hasMemory;

        return $this;
    }

    public function hasMemory(): bool
    {
        return $this->hasMemory;
    }

    /**
     * Get additional custom user data
     * @return array
     */
    public function getAdditional(): array
    {
        return $this->additional;
    }

    /**
     * Set additional custom user data
     * @param array $additional
     */
    public function setAdditional(array $additional): self
    {
        $this->additional = $additional;

        return $this;
    }
}
