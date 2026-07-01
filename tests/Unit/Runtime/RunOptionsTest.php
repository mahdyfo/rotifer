<?php

declare(strict_types=1);

namespace Rotifer\Tests\Unit\Runtime;

use PHPUnit\Framework\TestCase;
use Rotifer\Network\Activation\Tanh;
use Rotifer\Runtime\EvolutionConfig;
use Rotifer\Runtime\RunOptions;

/**
 * RunOptions is the single source of truth shared by the CLI flag parser, the
 * dashboard's defaults, and the dashboard's launch whitelist - so a knob added
 * here is settable from both the command line and the web by construction.
 */
final class RunOptionsTest extends TestCase
{
    public function testEverySchemaKeyHasADefaultAndViceVersa(): void
    {
        // Parity guard: the type schema (what may be overridden) and defaultsOf
        // (what the UI pre-fills) must describe exactly the same set of knobs.
        $this->assertSame(
            array_keys(RunOptions::TYPES),
            array_keys(RunOptions::defaultsOf(EvolutionConfig::default())),
        );
    }

    public function testStructuralOverridesApply(): void
    {
        $c = RunOptions::applyTo(EvolutionConfig::default(), [
            'seed' => '7', 'population' => '40', 'generations' => '9', 'islands' => '3',
            'activation' => 'tanh', 'hidden-layers' => '5,3', 'simplicity' => '2',
        ]);

        $this->assertSame(7, $c->getSeed());
        $this->assertSame(40, $c->getPopulation());
        $this->assertSame(9, $c->getGenerations());
        $this->assertSame(3, $c->getIslands());
        $this->assertInstanceOf(Tanh::class, $c->getActivation());
        $this->assertSame([5, 3], $c->getHiddenLayers());
        $this->assertSame(2, $c->getSimplicity());
    }

    public function testMemoryOverrideTogglesRecurrentState(): void
    {
        $on = RunOptions::applyTo(EvolutionConfig::default(), ['memory' => '1']);
        $this->assertTrue($on->hasMemory());

        $off = RunOptions::applyTo($on, ['memory' => '0']);
        $this->assertFalse($off->hasMemory());
    }

    public function testRandomWindowIsSettableAndDisablable(): void
    {
        $on = RunOptions::applyTo(EvolutionConfig::default(), ['window' => '5', 'window-prime' => '2']);
        $this->assertTrue($on->isRandomWindowEnabled());
        $this->assertSame(5, $on->getWindowSize());
        $this->assertSame(2, $on->getWindowPrime());

        // Explicit 0 turns the window off again.
        $off = RunOptions::applyTo($on, ['window' => '0']);
        $this->assertFalse($off->isRandomWindowEnabled());
    }

    public function testWeightMutationMechanicsAreSettable(): void
    {
        // The knob the user called out: weightMutation(count, adjustmentRange, randomizeProbability).
        $c = RunOptions::applyTo(EvolutionConfig::default(), [
            'weight-count' => '2', 'weight-adjust' => '0.8', 'weight-randomize' => '0.1',
        ]);

        $this->assertSame(2, $c->getWeightMutationCount());
        $this->assertSame(0.8, $c->getWeightAdjustmentRange());
        $this->assertSame(0.1, $c->getWeightRandomizeProbability());
    }

    public function testTraumaParametersAreSettable(): void
    {
        $c = RunOptions::applyTo(EvolutionConfig::default(), [
            'trauma' => '1', 'trauma-intensity' => '0.7', 'trauma-decay' => '0.2',
        ]);

        $this->assertTrue($c->isTraumaEnabled());
        $this->assertSame(0.7, $c->getTraumaIntensity());
        $this->assertSame(0.2, $c->getTraumaDecay());
    }

    public function testAdaptiveMutationParametersAreSettable(): void
    {
        $c = RunOptions::applyTo(EvolutionConfig::default(), [
            'adaptive-mutation' => '1', 'adaptive-patience' => '10',
            'adaptive-up' => '2.0', 'adaptive-down' => '0.8',
            'adaptive-min' => '0.5', 'adaptive-max' => '6.0',
        ]);

        $this->assertTrue($c->isAdaptiveMutationEnabled());
        $this->assertSame(10, $c->getAdaptivePatience());
        $this->assertSame(2.0, $c->getAdaptiveUpFactor());
        $this->assertSame(0.8, $c->getAdaptiveDownFactor());
        $this->assertSame(0.5, $c->getAdaptiveMinScale());
        $this->assertSame(6.0, $c->getAdaptiveMaxScale());
    }

    public function testLifetimeLearningParametersAreSettable(): void
    {
        $c = RunOptions::applyTo(EvolutionConfig::default(), [
            'lifetime-learning' => '1', 'lifetime-steps' => '8',
            'lifetime-step-size' => '0.4', 'lamarckian' => '0.5',
        ]);

        $this->assertTrue($c->isLifetimeLearningEnabled());
        $this->assertSame(8, $c->getLifetimeLearningSteps());
        $this->assertSame(0.4, $c->getLifetimeLearningStepSize());
        $this->assertSame(0.5, $c->getLamarckianFraction());
    }

    public function testBareLifetimeToggleKeepsTheEstablishedDefaults(): void
    {
        // Enabling lifetime learning with no explicit numbers (a bare CLI --lifetime-learning,
        // or the web toggle while the steps field is still 0) must do something useful.
        $c = RunOptions::applyTo(EvolutionConfig::default(), ['lifetime-learning' => '1']);

        $this->assertTrue($c->isLifetimeLearningEnabled());
        $this->assertSame(5, $c->getLifetimeLearningSteps());
        $this->assertSame(0.3, $c->getLamarckianFraction());
    }

    public function testDisablingLifetimeLearningWins(): void
    {
        $base = EvolutionConfig::default()->lifetimeLearning(steps: 5, lamarckian: 0.3);
        $c = RunOptions::applyTo($base, ['lifetime-learning' => '0', 'lifetime-steps' => '5']);

        $this->assertFalse($c->isLifetimeLearningEnabled());
    }

    public function testUnspecifiedOptionsLeaveTheConfigUntouched(): void
    {
        $base = EvolutionConfig::default()->population(321);
        $c = RunOptions::applyTo($base, []);

        $this->assertSame(321, $c->getPopulation());
        $this->assertFalse($c->isTraumaEnabled());
        $this->assertFalse($c->isAdaptiveMutationEnabled());
    }
}
