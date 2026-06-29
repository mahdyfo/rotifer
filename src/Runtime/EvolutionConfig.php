<?php

declare(strict_types=1);

namespace Rotifer\Runtime;

use Rotifer\Network\Activation\Activation;
use Rotifer\Network\Activation\Sigmoid;

/**
 * Every knob the engine reads, in one immutable value object - the replacement
 * for the legacy global constants (PROBABILITY_*, ACTIVATION, ...).
 *
 * Immutable by convention: each setter returns a modified clone, so a base
 * config can be shared and specialized without surprises, and the same config
 * can be rebuilt verbatim inside a parallel worker. Biological knobs are added
 * in their own phase; this phase covers structure and reproduction.
 */
final class EvolutionConfig
{
    private string $name = 'default';
    private int $seed = 1;

    // Population / structure
    private int $population = 150;
    private int $islands = 1;
    private int $generations = 50;
    private float $surviveRate = 0.5;
    private int $elitism = 1;
    private float $diversityInjection = 0.0;
    private bool $hasMemory = false;
    private int $initialHidden = 1;
    /** @var list<int> fixed hidden-layer sizes; empty = dynamic, evolving topology */
    private array $hiddenLayers = [];
    // significant figures of fitness treated as equal so simpler nets win on ties; 0 = off
    private int $simplicity = 3;
    private Activation $activation;

    // Reproduction
    private float $crossoverProbability = 0.5;
    private float $weightMutationProbability = 0.4;
    private float $addNeuronProbability = 0.03;
    private float $addConnectionProbability = 0.05;
    private float $removeNeuronProbability = 0.02;
    private float $removeConnectionProbability = 0.03;
    private int $weightMutationCount = 1;
    private float $weightAdjustmentRange = 0.5;
    private float $weightRandomizeProbability = 0.1;

    // Biology - each mechanism is independently optional; when off, its code path
    // is a skippable no-op, so complexity is strictly opt-in.
    private bool $traumaEnabled = false;
    private float $traumaIntensity = 0.4;
    private float $traumaDecay = 0.5;

    private bool $adaptiveMutationEnabled = false;
    private int $adaptivePatience = 6;
    private float $adaptiveUpFactor = 1.5;
    private float $adaptiveDownFactor = 0.95;
    private float $adaptiveMinScale = 0.25;
    private float $adaptiveMaxScale = 4.0;

    private bool $lifetimeLearningEnabled = false;
    private int $lifetimeLearningSteps = 0;
    private float $lifetimeLearningStepSize = 0.3;
    private float $lamarckianFraction = 0.0;

    private int $migrationEveryGenerations = 0;
    private int $migrationTopK = 1;

    // Parallelism is a yes/no choice: on evolves each island in its own worker
    // process (worker count = island count), off stays serial. This is only the
    // problem's *preference* - the CLI still needs --parallel and the dashboard
    // pre-checks its "parallel" box from it. A problem opts in with ->parallel().
    private bool $parallel = false;

    private function __construct()
    {
        $this->activation = new Sigmoid();
    }

    public static function default(): self
    {
        return new self();
    }

    // --- fluent setters (return a modified clone) ---

    public function name(string $name): self
    {
        $c = clone $this;
        $c->name = $name;
        return $c;
    }

    public function seed(int $seed): self
    {
        $c = clone $this;
        $c->seed = $seed;
        return $c;
    }

    public function population(int $population): self
    {
        $c = clone $this;
        $c->population = max(2, $population);
        return $c;
    }

    public function islands(int $islands): self
    {
        $c = clone $this;
        $c->islands = max(1, $islands);
        return $c;
    }

    public function generations(int $generations): self
    {
        $c = clone $this;
        $c->generations = max(0, $generations);
        return $c;
    }

    public function surviveRate(float $rate): self
    {
        $c = clone $this;
        $c->surviveRate = min(1.0, max(0.01, $rate));
        return $c;
    }

    public function elitism(int $count): self
    {
        $c = clone $this;
        $c->elitism = max(0, $count);
        return $c;
    }

    public function diversityInjection(float $rate): self
    {
        $c = clone $this;
        $c->diversityInjection = min(1.0, max(0.0, $rate));
        return $c;
    }

    public function memory(bool $hasMemory = true): self
    {
        $c = clone $this;
        $c->hasMemory = $hasMemory;
        return $c;
    }

    /** Number of hidden neurons each freshly seeded organism starts with. */
    public function initialHidden(int $count): self
    {
        $c = clone $this;
        $c->initialHidden = max(0, $count);
        return $c;
    }

    public function activation(Activation $activation): self
    {
        $c = clone $this;
        $c->activation = $activation;
        return $c;
    }

    // reward simpler networks on tied scores; $significantFigures sets the tolerance, 0 disables it
    public function simplicity(int $significantFigures): self
    {
        $c = clone $this;
        $c->simplicity = max(0, $significantFigures);
        return $c;
    }

    /**
     * Pin the network to a fixed, classic layered MLP with these hidden-layer sizes
     * (e.g. [3] for one bottleneck layer, [8, 4, 8] for three). Edges run only
     * between consecutive layers - no intra-layer or input->output shortcuts - and
     * the neuron count is frozen, so only the weights and wiring within those slots
     * evolve. An empty list (the default) keeps the dynamic, topology-evolving mode.
     *
     * @param list<int> $sizes
     */
    public function hiddenLayers(array $sizes): self
    {
        $c = clone $this;
        $c->hiddenLayers = array_values(array_filter(array_map(static fn ($n) => max(0, (int) $n), $sizes), static fn (int $n) => $n > 0));
        return $c;
    }

    public function crossover(float $probability): self
    {
        $c = clone $this;
        $c->crossoverProbability = $probability;
        return $c;
    }

    /**
     * Set the structural/weight mutation probabilities. Null leaves a rate
     * unchanged, so callers tweak only what they care about.
     */
    public function mutation(
        ?float $weight = null,
        ?float $addNeuron = null,
        ?float $addConnection = null,
        ?float $removeNeuron = null,
        ?float $removeConnection = null,
    ): self {
        $c = clone $this;
        $c->weightMutationProbability = $weight ?? $c->weightMutationProbability;
        $c->addNeuronProbability = $addNeuron ?? $c->addNeuronProbability;
        $c->addConnectionProbability = $addConnection ?? $c->addConnectionProbability;
        $c->removeNeuronProbability = $removeNeuron ?? $c->removeNeuronProbability;
        $c->removeConnectionProbability = $removeConnection ?? $c->removeConnectionProbability;
        return $c;
    }

    public function weightMutation(?int $count = null, ?float $adjustmentRange = null, ?float $randomizeProbability = null): self
    {
        $c = clone $this;
        $c->weightMutationCount = $count ?? $c->weightMutationCount;
        $c->weightAdjustmentRange = $adjustmentRange ?? $c->weightAdjustmentRange;
        $c->weightRandomizeProbability = $randomizeProbability ?? $c->weightRandomizeProbability;
        return $c;
    }

    // --- biology setters ---

    /** Heritable, decaying stress markers ("genetic trauma") that bias offspring. */
    public function trauma(bool $enabled = true, ?float $intensity = null, ?float $decay = null): self
    {
        $c = clone $this;
        $c->traumaEnabled = $enabled;
        $c->traumaIntensity = $intensity ?? $c->traumaIntensity;
        $c->traumaDecay = $decay ?? $c->traumaDecay;
        return $c;
    }

    /** Self-tuning mutation: explore harder when stuck, exploit when improving. */
    public function adaptiveMutation(
        bool $enabled = true,
        ?int $patience = null,
        ?float $up = null,
        ?float $down = null,
        ?float $minScale = null,
        ?float $maxScale = null,
    ): self {
        $c = clone $this;
        $c->adaptiveMutationEnabled = $enabled;
        $c->adaptivePatience = $patience ?? $c->adaptivePatience;
        $c->adaptiveUpFactor = $up ?? $c->adaptiveUpFactor;
        $c->adaptiveDownFactor = $down ?? $c->adaptiveDownFactor;
        $c->adaptiveMinScale = $minScale ?? $c->adaptiveMinScale;
        $c->adaptiveMaxScale = $maxScale ?? $c->adaptiveMaxScale;
        return $c;
    }

    /**
     * Within-lifetime weight refinement (Baldwin). $lamarckian in [0,1] is the
     * fraction of what was learned that is written back into the genome and
     * therefore inherited.
     */
    public function lifetimeLearning(int $steps = 5, float $lamarckian = 0.0, ?float $stepSize = null, bool $enabled = true): self
    {
        $c = clone $this;
        $c->lifetimeLearningEnabled = $enabled;
        $c->lifetimeLearningSteps = max(0, $steps);
        $c->lamarckianFraction = min(1.0, max(0.0, $lamarckian));
        $c->lifetimeLearningStepSize = $stepSize ?? $c->lifetimeLearningStepSize;
        return $c;
    }

    /** Move the top $topK organisms between neighbouring islands every N generations. */
    public function migration(int $everyGenerations, int $topK = 1): self
    {
        $c = clone $this;
        $c->migrationEveryGenerations = max(0, $everyGenerations);
        $c->migrationTopK = max(1, $topK);
        return $c;
    }

    /** Whether this problem prefers a parallel run (island-per-worker); see {@see $parallel}. */
    public function parallel(bool $enabled = true): self
    {
        $c = clone $this;
        $c->parallel = $enabled;
        return $c;
    }

    // --- getters ---

    public function getName(): string
    {
        return $this->name;
    }

    public function getSeed(): int
    {
        return $this->seed;
    }

    public function getPopulation(): int
    {
        return $this->population;
    }

    public function getIslands(): int
    {
        return $this->islands;
    }

    public function getGenerations(): int
    {
        return $this->generations;
    }

    public function getSurviveRate(): float
    {
        return $this->surviveRate;
    }

    public function getElitism(): int
    {
        return $this->elitism;
    }

    public function getDiversityInjection(): float
    {
        return $this->diversityInjection;
    }

    public function hasMemory(): bool
    {
        return $this->hasMemory;
    }

    public function getInitialHidden(): int
    {
        return $this->initialHidden;
    }

    /** @return list<int> fixed hidden-layer sizes; empty means dynamic topology */
    public function getHiddenLayers(): array
    {
        return $this->hiddenLayers;
    }

    public function getSimplicity(): int
    {
        return $this->simplicity;
    }

    public function getActivation(): Activation
    {
        return $this->activation;
    }

    public function getCrossoverProbability(): float
    {
        return $this->crossoverProbability;
    }

    public function getWeightMutationProbability(): float
    {
        return $this->weightMutationProbability;
    }

    public function getAddNeuronProbability(): float
    {
        return $this->addNeuronProbability;
    }

    public function getAddConnectionProbability(): float
    {
        return $this->addConnectionProbability;
    }

    public function getRemoveNeuronProbability(): float
    {
        return $this->removeNeuronProbability;
    }

    public function getRemoveConnectionProbability(): float
    {
        return $this->removeConnectionProbability;
    }

    public function getWeightMutationCount(): int
    {
        return $this->weightMutationCount;
    }

    public function getWeightAdjustmentRange(): float
    {
        return $this->weightAdjustmentRange;
    }

    public function getWeightRandomizeProbability(): float
    {
        return $this->weightRandomizeProbability;
    }

    public function isTraumaEnabled(): bool
    {
        return $this->traumaEnabled;
    }

    public function getTraumaIntensity(): float
    {
        return $this->traumaIntensity;
    }

    public function getTraumaDecay(): float
    {
        return $this->traumaDecay;
    }

    public function isAdaptiveMutationEnabled(): bool
    {
        return $this->adaptiveMutationEnabled;
    }

    public function getAdaptivePatience(): int
    {
        return $this->adaptivePatience;
    }

    public function getAdaptiveUpFactor(): float
    {
        return $this->adaptiveUpFactor;
    }

    public function getAdaptiveDownFactor(): float
    {
        return $this->adaptiveDownFactor;
    }

    public function getAdaptiveMinScale(): float
    {
        return $this->adaptiveMinScale;
    }

    public function getAdaptiveMaxScale(): float
    {
        return $this->adaptiveMaxScale;
    }

    public function isLifetimeLearningEnabled(): bool
    {
        return $this->lifetimeLearningEnabled && $this->lifetimeLearningSteps > 0;
    }

    public function getLifetimeLearningSteps(): int
    {
        return $this->lifetimeLearningSteps;
    }

    public function getLifetimeLearningStepSize(): float
    {
        return $this->lifetimeLearningStepSize;
    }

    public function getLamarckianFraction(): float
    {
        return $this->lamarckianFraction;
    }

    public function getMigrationEveryGenerations(): int
    {
        return $this->migrationEveryGenerations;
    }

    public function getMigrationTopK(): int
    {
        return $this->migrationTopK;
    }

    /** Whether this problem prefers a parallel run (pre-checks the dashboard's box). */
    public function isParallelEnabled(): bool
    {
        return $this->parallel;
    }
}
