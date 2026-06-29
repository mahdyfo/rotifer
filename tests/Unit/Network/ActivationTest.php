<?php

declare(strict_types=1);

namespace Rotifer\Tests\Unit\Network;

use PHPUnit\Framework\TestCase;
use Rotifer\Network\Activation\ActivationFactory;
use Rotifer\Network\Activation\Gelu;
use Rotifer\Network\Activation\LeakyRelu;
use Rotifer\Network\Activation\Relu;
use Rotifer\Network\Activation\Sigmoid;
use Rotifer\Network\Activation\Threshold;

final class ActivationTest extends TestCase
{
    public function testSigmoidCentersAtHalf(): void
    {
        $this->assertEqualsWithDelta(0.5, (new Sigmoid())->activate(0.0), 1e-9);
        $this->assertGreaterThan(0.7, (new Sigmoid())->activate(1.0));
    }

    public function testReluClipsNegatives(): void
    {
        $relu = new Relu();
        $this->assertSame(0.0, $relu->activate(-3.0));
        $this->assertSame(2.5, $relu->activate(2.5));
    }

    public function testLeakyReluLeaks(): void
    {
        $leaky = new LeakyRelu(0.01);
        $this->assertEqualsWithDelta(-0.05, $leaky->activate(-5.0), 1e-9);
        $this->assertSame(5.0, $leaky->activate(5.0));
    }

    public function testThresholdIsBinary(): void
    {
        $threshold = new Threshold();
        $this->assertSame(0.0, $threshold->activate(-0.0001));
        $this->assertSame(1.0, $threshold->activate(0.0));
        $this->assertSame(1.0, $threshold->activate(10.0));
    }

    public function testGeluIsNearZeroAtZeroAndPassesLargePositives(): void
    {
        $gelu = new Gelu();
        $this->assertEqualsWithDelta(0.0, $gelu->activate(0.0), 1e-9);
        $this->assertGreaterThan(2.9, $gelu->activate(3.0));
    }

    public function testFactoryRebuildsByName(): void
    {
        foreach (['sigmoid', 'relu', 'leaky_relu', 'tanh', 'threshold', 'gelu', 'softmax'] as $name) {
            $this->assertSame($name, ActivationFactory::fromName($name)->name());
        }
    }
}
