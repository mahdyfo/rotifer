<?php

namespace Rotifer\Tests\Unit\Helpers;

use PHPUnit\Framework\TestCase;
use Rotifer\Helpers\WeightHelper;

class WeightHelperTest extends TestCase
{
    public function testMaxWeightConstant(): void
    {
        $this->assertEquals(8.388607, WeightHelper::MAX_WEIGHT);
    }

    public function testGenerateRandomWeightReturnsFloat(): void
    {
        $weight = WeightHelper::generateRandomWeight();
        $this->assertIsFloat($weight);
    }

    public function testGenerateRandomWeightWithinBounds(): void
    {
        for ($i = 0; $i < 100; $i++) {
            $weight = WeightHelper::generateRandomWeight();

            $this->assertGreaterThanOrEqual(-WeightHelper::MAX_WEIGHT, $weight);
            $this->assertLessThanOrEqual(WeightHelper::MAX_WEIGHT, $weight);
        }
    }

    public function testGenerateRandomWeightDistribution(): void
    {
        $weights = [];

        for ($i = 0; $i < 1000; $i++) {
            $weights[] = WeightHelper::generateRandomWeight();
        }

        $positiveCount = count(array_filter($weights, fn($w) => $w > 0));
        $negativeCount = count(array_filter($weights, fn($w) => $w < 0));
        $zeroCount = count(array_filter($weights, fn($w) => $w == 0));

        // Check that we have both positive and negative weights
        $this->assertGreaterThan(0, $positiveCount);
        $this->assertGreaterThan(0, $negativeCount);

        // Distribution should be roughly balanced (within 40-60% range)
        $positiveRatio = $positiveCount / 1000;
        $this->assertGreaterThan(0.3, $positiveRatio);
        $this->assertLessThan(0.7, $positiveRatio);
    }

    public function testGenerateRandomWeightPrecision(): void
    {
        $weight = WeightHelper::generateRandomWeight();

        // Weight should have up to 6 decimal places of precision
        $decimalPlaces = strlen(substr(strrchr((string)$weight, "."), 1));
        $this->assertLessThanOrEqual(6, $decimalPlaces);
    }

    public function testMultipleCallsProduceDifferentValues(): void
    {
        $weight1 = WeightHelper::generateRandomWeight();
        $weight2 = WeightHelper::generateRandomWeight();
        $weight3 = WeightHelper::generateRandomWeight();

        // At least one should be different (extremely unlikely all three are identical)
        $allSame = ($weight1 === $weight2) && ($weight2 === $weight3);
        $this->assertFalse($allSame);
    }

    public function testGenerateWeightNearBoundaries(): void
    {
        // Generate many weights and check if some are near boundaries
        $nearMax = false;
        $nearMin = false;

        for ($i = 0; $i < 10000; $i++) {
            $weight = WeightHelper::generateRandomWeight();

            if ($weight > WeightHelper::MAX_WEIGHT * 0.9) {
                $nearMax = true;
            }
            if ($weight < -WeightHelper::MAX_WEIGHT * 0.9) {
                $nearMin = true;
            }

            if ($nearMax && $nearMin) {
                break;
            }
        }

        $this->assertTrue($nearMax, 'Should generate weights near maximum');
        $this->assertTrue($nearMin, 'Should generate weights near minimum');
    }
}
