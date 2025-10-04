<?php

namespace Rotifer\Tests\Unit\Activations;

use PHPUnit\Framework\TestCase;
use Rotifer\Activations\Activation;

class ActivationTest extends TestCase
{
    public function testSigmoidWithZero(): void
    {
        $result = Activation::sigmoid(0);
        $this->assertEqualsWithDelta(0.5, $result, 0.0001);
    }

    public function testSigmoidWithPositiveValue(): void
    {
        $result = Activation::sigmoid(2);
        // sigmoid(2) ≈ 0.8808
        $this->assertEqualsWithDelta(0.8808, $result, 0.001);
        $this->assertGreaterThan(0.5, $result);
        $this->assertLessThan(1.0, $result);
    }

    public function testSigmoidWithNegativeValue(): void
    {
        $result = Activation::sigmoid(-2);
        // sigmoid(-2) ≈ 0.1192
        $this->assertEqualsWithDelta(0.1192, $result, 0.001);
        $this->assertLessThan(0.5, $result);
        $this->assertGreaterThan(0.0, $result);
    }

    public function testSigmoidWithLargePositiveValue(): void
    {
        $result = Activation::sigmoid(10);
        // sigmoid(10) ≈ 0.9999
        $this->assertGreaterThan(0.999, $result);
        $this->assertLessThan(1.0, $result);
    }

    public function testSigmoidWithLargeNegativeValue(): void
    {
        $result = Activation::sigmoid(-10);
        // sigmoid(-10) ≈ 0.0001
        $this->assertLessThan(0.001, $result);
        $this->assertGreaterThan(0.0, $result);
    }

    public function testSigmoidRange(): void
    {
        // Test that sigmoid always returns values between 0 and 1
        $testValues = [-100, -10, -5, -1, 0, 1, 5, 10, 100];

        foreach ($testValues as $value) {
            $result = Activation::sigmoid($value);
            $this->assertGreaterThan(0, $result);
            $this->assertLessThanOrEqual(1, $result);
        }
    }

    public function testReluWithPositiveValue(): void
    {
        $result = Activation::relu(5.0);
        $this->assertEquals(5.0, $result);
    }

    public function testReluWithNegativeValue(): void
    {
        $result = Activation::relu(-3.0);
        $this->assertEquals(0.0, $result);
    }

    public function testReluWithZero(): void
    {
        $result = Activation::relu(0.0);
        $this->assertEquals(0.0, $result);
    }

    public function testReluWithLargeValues(): void
    {
        $this->assertEquals(1000.0, Activation::relu(1000.0));
        $this->assertEquals(0.0, Activation::relu(-1000.0));
    }

    public function testLeakyReluWithPositiveValue(): void
    {
        $result = Activation::leakyRelu(5.0);
        $this->assertEquals(5.0, $result);
    }

    public function testLeakyReluWithNegativeValue(): void
    {
        $result = Activation::leakyRelu(-10.0);
        // LeakyReLU(-10) = -10 * 0.01 = -0.1
        $this->assertEquals(-0.1, $result);
    }

    public function testLeakyReluWithZero(): void
    {
        $result = Activation::leakyRelu(0.0);
        $this->assertEquals(0.0, $result);
    }

    public function testLeakyReluNeverFullyZero(): void
    {
        // LeakyReLU should return small negative values instead of zero
        $result = Activation::leakyRelu(-5.0);
        $this->assertEquals(-0.05, $result);
        $this->assertNotSame(0.0, $result);
    }

    public function testTanhWithZero(): void
    {
        $result = Activation::tanh(0);
        $this->assertEquals(0.0, $result);
    }

    public function testTanhWithPositiveValue(): void
    {
        $result = Activation::tanh(4);
        // tanh(4 * 0.25) = tanh(1) ≈ 0.7616
        $this->assertEqualsWithDelta(0.7616, $result, 0.001);
    }

    public function testTanhWithNegativeValue(): void
    {
        $result = Activation::tanh(-4);
        // tanh(-4 * 0.25) = tanh(-1) ≈ -0.7616
        $this->assertEqualsWithDelta(-0.7616, $result, 0.001);
    }

    public function testTanhRange(): void
    {
        // tanh always returns values between -1 and 1
        $testValues = [-100, -10, -1, 0, 1, 10, 100];

        foreach ($testValues as $value) {
            $result = Activation::tanh($value);
            $this->assertGreaterThanOrEqual(-1.0, $result);
            $this->assertLessThanOrEqual(1.0, $result);
        }
    }

    public function testTanhWithCustomMultiplication(): void
    {
        $result = Activation::tanh(4, 0.5);
        // tanh(4 * 0.5) = tanh(2) ≈ 0.9640
        $this->assertEqualsWithDelta(0.9640, $result, 0.001);
    }

    public function testThresholdWithPositiveValue(): void
    {
        $result = Activation::threshold(5.0);
        $this->assertEquals(1, $result);
    }

    public function testThresholdWithNegativeValue(): void
    {
        $result = Activation::threshold(-3.0);
        $this->assertEquals(0, $result);
    }

    public function testThresholdWithZero(): void
    {
        $result = Activation::threshold(0.0);
        $this->assertEquals(1, $result);
    }

    public function testThresholdIsBinary(): void
    {
        $testValues = [-100, -10, -0.1, 0, 0.1, 10, 100];

        foreach ($testValues as $value) {
            $result = Activation::threshold($value);
            $this->assertTrue($result === 0.0 || $result === 1.0 || $result === 0 || $result === 1);
        }
    }

    public function testAllActivationsReturnNumeric(): void
    {
        $value = 2.5;

        $this->assertIsFloat(Activation::sigmoid($value));
        $this->assertIsFloat(Activation::relu($value));
        $this->assertIsFloat(Activation::leakyRelu($value));
        $this->assertIsFloat(Activation::tanh($value));
        $this->assertIsFloat(Activation::threshold($value));
    }

    public function testActivationSymmetry(): void
    {
        $value = 3.0;

        // Sigmoid and tanh should be symmetric around zero
        $sigmoidPos = Activation::sigmoid($value);
        $sigmoidNeg = Activation::sigmoid(-$value);
        $this->assertEqualsWithDelta(1 - $sigmoidPos, $sigmoidNeg, 0.0001);

        $tanhPos = Activation::tanh($value);
        $tanhNeg = Activation::tanh(-$value);
        $this->assertEqualsWithDelta(-$tanhPos, $tanhNeg, 0.0001);
    }

    public function testActivationMonotonicity(): void
    {
        // Sigmoid, relu, leakyRelu and tanh should be monotonically increasing
        // For monotonically increasing: f(1) < f(2)
        $this->assertLessThan(Activation::sigmoid(2), Activation::sigmoid(1));
        $this->assertLessThan(Activation::relu(2), Activation::relu(1));
        $this->assertLessThan(Activation::leakyRelu(2), Activation::leakyRelu(1));
        $this->assertLessThan(Activation::tanh(2), Activation::tanh(1));
    }
}
