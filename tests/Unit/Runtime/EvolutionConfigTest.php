<?php

declare(strict_types=1);

namespace Rotifer\Tests\Unit\Runtime;

use PHPUnit\Framework\TestCase;
use Rotifer\Network\Activation\Sigmoid;
use Rotifer\Network\Activation\Tanh;
use Rotifer\Runtime\EvolutionConfig;

final class EvolutionConfigTest extends TestCase
{
    public function testSensibleDefaults(): void
    {
        $config = EvolutionConfig::default();
        $this->assertSame('default', $config->getName());
        $this->assertSame(1, $config->getIslands());
        $this->assertInstanceOf(Sigmoid::class, $config->getActivation());
    }

    public function testSettersReturnAModifiedCloneWithoutMutatingTheOriginal(): void
    {
        $base = EvolutionConfig::default();
        $tuned = $base->population(300)->islands(4)->seed(99);

        $this->assertSame(150, $base->getPopulation(), 'original untouched');
        $this->assertSame(1, $base->getIslands());

        $this->assertSame(300, $tuned->getPopulation());
        $this->assertSame(4, $tuned->getIslands());
        $this->assertSame(99, $tuned->getSeed());
    }

    public function testMutationLeavesUnspecifiedRatesUnchanged(): void
    {
        $config = EvolutionConfig::default()
            ->mutation(weight: 0.9)
            ->mutation(addNeuron: 0.1);

        $this->assertSame(0.9, $config->getWeightMutationProbability());
        $this->assertSame(0.1, $config->getAddNeuronProbability());
    }

    public function testActivationIsReplaceable(): void
    {
        $config = EvolutionConfig::default()->activation(new Tanh());
        $this->assertInstanceOf(Tanh::class, $config->getActivation());
    }

    public function testPopulationFloorIsTwo(): void
    {
        $this->assertSame(2, EvolutionConfig::default()->population(1)->getPopulation());
    }
}
