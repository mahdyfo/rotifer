<?php

declare(strict_types=1);

namespace Rotifer\Runtime;

use Rotifer\Network\Activation\ActivationFactory;

/**
 * The one place that knows every run-level override an EvolutionConfig exposes.
 *
 * Three consumers share it, which is what keeps the command line and the web
 * dashboard at parity: the CLI flag parser ({@see \Rotifer\Cli\Console}) calls
 * {@see applyTo()} to fold flags into a config; the dashboard pre-fills its form
 * from {@see defaultsOf()}; and the dashboard's launcher
 * ({@see \Rotifer\Web\RunManager}) whitelists exactly {@see TYPES} before
 * shelling out to the same flags. Add a knob here and it is settable from both.
 *
 * Keys are the kebab-case flag names (`--weight-count`, the `weight-count`
 * override field). `parallel` and `resume` are deliberately absent: they steer
 * the engine/launcher, not the config, and are handled by their own callers.
 */
final class RunOptions
{
    /** Every override key => its scalar type (`int`, `float`, `bool`, `string`). */
    public const TYPES = [
        // Population / structure
        'seed' => 'int',
        'population' => 'int',
        'generations' => 'int',
        'islands' => 'int',
        'survive-rate' => 'float',
        'elitism' => 'int',
        'diversity' => 'float',
        'initial-hidden' => 'int',
        'hidden-layers' => 'string',
        'simplicity' => 'int',
        'activation' => 'string',
        // Reproduction
        'crossover' => 'float',
        'weight-mutation' => 'float',
        'add-neuron' => 'float',
        'add-connection' => 'float',
        'remove-neuron' => 'float',
        'remove-connection' => 'float',
        'weight-count' => 'int',
        'weight-adjust' => 'float',
        'weight-randomize' => 'float',
        // Biology: trauma
        'trauma' => 'bool',
        'trauma-intensity' => 'float',
        'trauma-decay' => 'float',
        // Biology: adaptive mutation
        'adaptive-mutation' => 'bool',
        'adaptive-patience' => 'int',
        'adaptive-up' => 'float',
        'adaptive-down' => 'float',
        'adaptive-min' => 'float',
        'adaptive-max' => 'float',
        // Biology: lifetime learning
        'lifetime-learning' => 'bool',
        'lifetime-steps' => 'int',
        'lifetime-step-size' => 'float',
        'lamarckian' => 'float',
        // Migration
        'migration-every' => 'int',
        'migration-top' => 'int',
    ];

    /** Fold a set of overrides (CLI flags or dashboard fields) into a config.
     *
     * Each option is applied only when present, so callers tweak just what they
     * care about and everything else keeps the problem's own configured value.
     *
     * @param array<string, scalar> $options
     */
    public static function applyTo(EvolutionConfig $config, array $options): EvolutionConfig
    {
        $int = static fn (string $k): ?int => isset($options[$k]) ? (int) $options[$k] : null;
        $float = static fn (string $k): ?float => isset($options[$k]) ? (float) $options[$k] : null;

        if (isset($options['seed'])) {
            $config = $config->seed((int) $options['seed']);
        }
        if (isset($options['population'])) {
            $config = $config->population((int) $options['population']);
        }
        if (isset($options['generations'])) {
            $config = $config->generations((int) $options['generations']);
        }
        if (isset($options['islands'])) {
            $config = $config->islands((int) $options['islands']);
        }
        if (isset($options['survive-rate'])) {
            $config = $config->surviveRate((float) $options['survive-rate']);
        }
        if (isset($options['elitism'])) {
            $config = $config->elitism((int) $options['elitism']);
        }
        if (isset($options['diversity'])) {
            $config = $config->diversityInjection((float) $options['diversity']);
        }
        if (isset($options['initial-hidden'])) {
            $config = $config->initialHidden((int) $options['initial-hidden']);
        }
        if (isset($options['simplicity'])) {
            $config = $config->simplicity((int) $options['simplicity']);
        }
        // "5,3,5" pins a fixed layered network; "" (or only zeros) forces dynamic topology.
        if (isset($options['hidden-layers']) && is_string($options['hidden-layers'])) {
            $sizes = array_values(array_filter(
                array_map('intval', explode(',', $options['hidden-layers'])),
                static fn (int $n): bool => $n > 0,
            ));
            $config = $config->hiddenLayers($sizes);
        }
        if (isset($options['activation'])) {
            $config = $config->activation(ActivationFactory::fromName((string) $options['activation']));
        }
        if (isset($options['crossover'])) {
            $config = $config->crossover((float) $options['crossover']);
        }

        // Mutation probabilities and the weight-nudge mechanics: null leaves a rate unchanged.
        $config = $config->mutation(
            weight: $float('weight-mutation'),
            addNeuron: $float('add-neuron'),
            addConnection: $float('add-connection'),
            removeNeuron: $float('remove-neuron'),
            removeConnection: $float('remove-connection'),
        );
        $config = $config->weightMutation(
            count: $int('weight-count'),
            adjustmentRange: $float('weight-adjust'),
            randomizeProbability: $float('weight-randomize'),
        );

        // Biology mechanisms: each touched only when its toggle or one of its
        // parameters is present, so an untouched mechanism keeps its config state.
        if (self::touches($options, ['trauma', 'trauma-intensity', 'trauma-decay'])) {
            $config = $config->trauma(
                array_key_exists('trauma', $options) ? self::bool($options['trauma']) : $config->isTraumaEnabled(),
                $float('trauma-intensity'),
                $float('trauma-decay'),
            );
        }
        if (self::touches($options, ['adaptive-mutation', 'adaptive-patience', 'adaptive-up', 'adaptive-down', 'adaptive-min', 'adaptive-max'])) {
            $config = $config->adaptiveMutation(
                array_key_exists('adaptive-mutation', $options) ? self::bool($options['adaptive-mutation']) : $config->isAdaptiveMutationEnabled(),
                $int('adaptive-patience'),
                $float('adaptive-up'),
                $float('adaptive-down'),
                $float('adaptive-min'),
                $float('adaptive-max'),
            );
        }
        if (self::touches($options, ['lifetime-learning', 'lifetime-steps', 'lifetime-step-size', 'lamarckian'])) {
            $enabled = array_key_exists('lifetime-learning', $options)
                ? self::bool($options['lifetime-learning'])
                : $config->isLifetimeLearningEnabled();
            if ($enabled) {
                // Enabling with no usable step count (a bare toggle, or the web's 0)
                // falls back to the established default so the switch actually does something.
                $steps = $int('lifetime-steps');
                if ($steps === null || $steps <= 0) {
                    $steps = $config->getLifetimeLearningSteps() ?: 5;
                }
                $lamarckian = $float('lamarckian') ?? ($config->getLamarckianFraction() ?: 0.3);
                $config = $config->lifetimeLearning(steps: $steps, lamarckian: $lamarckian, stepSize: $float('lifetime-step-size'), enabled: true);
            } else {
                $config = $config->lifetimeLearning(steps: 0, enabled: false);
            }
        }

        if (isset($options['migration-every'])) {
            $config = $config->migration(
                (int) $options['migration-every'],
                isset($options['migration-top']) ? (int) $options['migration-top'] : $config->getMigrationTopK(),
            );
        }

        return $config;
    }

    /**
     * The current value of every knob, keyed by override name - what the dashboard
     * pre-fills its form with so a run reproduces the problem's own config unless edited.
     *
     * @return array<string, scalar>
     */
    public static function defaultsOf(EvolutionConfig $c): array
    {
        return [
            'seed' => $c->getSeed(),
            'population' => $c->getPopulation(),
            'generations' => $c->getGenerations(),
            'islands' => $c->getIslands(),
            'survive-rate' => $c->getSurviveRate(),
            'elitism' => $c->getElitism(),
            'diversity' => $c->getDiversityInjection(),
            'initial-hidden' => $c->getInitialHidden(),
            'hidden-layers' => implode(',', $c->getHiddenLayers()),
            'simplicity' => $c->getSimplicity(),
            'activation' => $c->getActivation()->name(),
            'crossover' => $c->getCrossoverProbability(),
            'weight-mutation' => $c->getWeightMutationProbability(),
            'add-neuron' => $c->getAddNeuronProbability(),
            'add-connection' => $c->getAddConnectionProbability(),
            'remove-neuron' => $c->getRemoveNeuronProbability(),
            'remove-connection' => $c->getRemoveConnectionProbability(),
            'weight-count' => $c->getWeightMutationCount(),
            'weight-adjust' => $c->getWeightAdjustmentRange(),
            'weight-randomize' => $c->getWeightRandomizeProbability(),
            'trauma' => $c->isTraumaEnabled(),
            'trauma-intensity' => $c->getTraumaIntensity(),
            'trauma-decay' => $c->getTraumaDecay(),
            'adaptive-mutation' => $c->isAdaptiveMutationEnabled(),
            'adaptive-patience' => $c->getAdaptivePatience(),
            'adaptive-up' => $c->getAdaptiveUpFactor(),
            'adaptive-down' => $c->getAdaptiveDownFactor(),
            'adaptive-min' => $c->getAdaptiveMinScale(),
            'adaptive-max' => $c->getAdaptiveMaxScale(),
            'lifetime-learning' => $c->isLifetimeLearningEnabled(),
            'lifetime-steps' => $c->getLifetimeLearningSteps(),
            'lifetime-step-size' => $c->getLifetimeLearningStepSize(),
            'lamarckian' => $c->getLamarckianFraction(),
            'migration-every' => $c->getMigrationEveryGenerations(),
            'migration-top' => $c->getMigrationTopK(),
        ];
    }

    /** @param array<string, scalar> $options @param list<string> $keys */
    private static function touches(array $options, array $keys): bool
    {
        foreach ($keys as $key) {
            if (array_key_exists($key, $options)) {
                return true;
            }
        }
        return false;
    }

    private static function bool(mixed $value): bool
    {
        return $value === true ? true : filter_var($value, FILTER_VALIDATE_BOOLEAN);
    }
}
