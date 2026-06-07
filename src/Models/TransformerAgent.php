<?php

namespace Rotifer\Models;

use Exception;
use Rotifer\Activations\Activation;

/**
 * TransformerAgent - A specialized agent that implements transformer-like attention mechanisms
 *
 * Architecture:
 * - Input layer: receives token embeddings
 * - Attention layer: neurons compute attention scores (Query, Key, Value)
 * - Hidden layer: standard processing with evolved connections
 * - Output layer: produces predictions
 *
 * Unlike traditional transformers, this uses Rotifer's genetic evolution to discover
 * optimal attention patterns rather than learning them via backpropagation.
 */
class TransformerAgent extends Agent
{
    private int $attentionHeads = 2;
    private int $sequenceLength = 0;
    private array $attentionMemory = []; // Stores attention context

    public function setAttentionHeads(int $heads): self
    {
        $this->attentionHeads = $heads;
        return $this;
    }

    public function getAttentionHeads(): int
    {
        return $this->attentionHeads;
    }

    /**
     * Override step to include attention mechanism
     */
    public function step(array $inputValues): static
    {
        // Store input in attention memory for context
        $this->attentionMemory[] = $inputValues;

        // Limit attention memory to last N inputs (sequence length)
        $maxContextLength = 10; // Can be made configurable
        if (count($this->attentionMemory) > $maxContextLength) {
            array_shift($this->attentionMemory);
        }

        // Compute attention-weighted inputs
        $attentionWeightedInput = $this->computeAttention($inputValues);

        // Use attention-weighted input for standard forward pass
        return parent::step($attentionWeightedInput);
    }

    /**
     * Compute attention over input sequence
     * This is a simplified attention mechanism that works within Rotifer's architecture
     */
    private function computeAttention(array $currentInput): array
    {
        if (empty($this->attentionMemory)) {
            return $currentInput;
        }

        // Simplified multi-head attention
        $attendedInput = $currentInput;

        // For each attention head, compute a different attention pattern
        for ($head = 0; $head < $this->attentionHeads; $head++) {
            $headOffset = $head * 0.5; // Each head has slightly different scaling

            // Compute attention scores for each position in memory
            $attentionScores = [];
            $totalScore = 0;

            foreach ($this->attentionMemory as $memIdx => $memInput) {
                // Simple dot-product attention (query · key)
                $score = 0;
                $minLen = min(count($currentInput), count($memInput));

                for ($i = 0; $i < $minLen; $i++) {
                    // Add head-specific variation
                    $queryWeight = 1.0 + $headOffset;
                    $keyWeight = 1.0 - $headOffset * 0.5;
                    $score += ($currentInput[$i] * $queryWeight) * ($memInput[$i] * $keyWeight);
                }

                // Apply softmax-like normalization
                $score = exp($score / sqrt($minLen));
                $attentionScores[$memIdx] = $score;
                $totalScore += $score;
            }

            // Normalize scores
            if ($totalScore > 0) {
                foreach ($attentionScores as $idx => $score) {
                    $attentionScores[$idx] = $score / $totalScore;
                }
            }

            // Compute weighted sum of values
            foreach ($this->attentionMemory as $memIdx => $memInput) {
                $weight = $attentionScores[$memIdx] ?? 0;
                for ($i = 0; $i < count($attendedInput); $i++) {
                    if (isset($memInput[$i])) {
                        // Blend current input with attended memory
                        $attendedInput[$i] += $memInput[$i] * $weight * 0.3; // 0.3 = attention influence
                    }
                }
            }
        }

        return $attendedInput;
    }

    /**
     * Override reset to clear attention memory
     */
    public function reset(): static
    {
        $this->attentionMemory = [];
        return parent::reset();
    }

    /**
     * Override resetMemory to also clear attention context
     */
    public function resetMemory(): static
    {
        $this->attentionMemory = [];
        return parent::resetMemory();
    }

    /**
     * Create a transformer agent with attention layers
     */
    public static function createTransformer(
        int $inputSize,
        int $hiddenSize,
        int $outputSize,
        int $attentionHeads = 2,
        bool $hasMemory = true
    ): self {
        $agent = new self();
        $agent->createNeuron(Neuron::TYPE_INPUT, $inputSize);
        $agent->createNeuron(Neuron::TYPE_HIDDEN, $hiddenSize);
        $agent->createNeuron(Neuron::TYPE_OUTPUT, $outputSize);
        $agent->setHasMemory($hasMemory);
        $agent->setAttentionHeads($attentionHeads);
        $agent->initRandomConnections();

        // Add extra connections for attention-like patterns
        $agent->initAttentionConnections();

        return $agent;
    }

    /**
     * Initialize additional connections that support attention-like computation
     */
    private function initAttentionConnections(): self
    {
        $hiddenNeurons = $this->getNeuronsByType(Neuron::TYPE_HIDDEN);

        // Create more recurrent connections for attention mechanism
        if ($this->hasMemory() && count($hiddenNeurons) > 1) {
            $hiddenArray = array_values($hiddenNeurons);

            // Create query-key-value-like connections between hidden neurons
            for ($i = 0; $i < count($hiddenArray); $i++) {
                for ($j = 0; $j < count($hiddenArray); $j++) {
                    if ($i !== $j) {
                        // Create bidirectional connections for attention
                        $weight = \Rotifer\Helpers\WeightHelper::generateRandomWeight() * 0.5;
                        $this->connectNeurons($hiddenArray[$i], $hiddenArray[$j], $weight);
                    }
                }
            }
        }

        return $this;
    }

    /**
     * Get the attention memory (for debugging/visualization)
     */
    public function getAttentionMemory(): array
    {
        return $this->attentionMemory;
    }
}
