<?php

declare(strict_types=1);

namespace Rotifer\Tests\Unit\Runtime;

use PHPUnit\Framework\TestCase;
use Rotifer\Runtime\EvolutionConfig;
use Rotifer\Runtime\Fitness\WindowSelector;

final class WindowSelectorTest extends TestCase
{
    public function testDisabledConfigYieldsNoSelector(): void
    {
        $this->assertNull(WindowSelector::fromConfig(EvolutionConfig::default()));
        $this->assertNull(WindowSelector::fromConfig(EvolutionConfig::default()->randomWindow(0)));
    }

    public function testConfigTogglesAndStoresWindow(): void
    {
        $on = EvolutionConfig::default()->randomWindow(5);
        $this->assertTrue($on->isRandomWindowEnabled());
        $this->assertSame(5, $on->getWindowSize());
        $this->assertSame(0, $on->getWindowPrime());

        $primed = EvolutionConfig::default()->randomWindow(4, prime: 2);
        $this->assertSame(2, $primed->getWindowPrime());
        $this->assertSame(4, $primed->getWindowSize());

        $this->assertFalse(EvolutionConfig::default()->randomWindow(0)->isRandomWindowEnabled());
    }

    public function testNoWindowWhenItWouldCoverEveryRow(): void
    {
        $selector = new WindowSelector(seed: 1, windowSize: 5, prime: 0);
        $this->assertNull($selector->forGeneration(1, 5), 'window == rows: nothing to slide');
        $this->assertNull($selector->forGeneration(1, 3), 'window > rows');
    }

    public function testWindowStaysInRangeAndIsDeterministic(): void
    {
        $a = new WindowSelector(seed: 42, windowSize: 5, prime: 0);
        $b = new WindowSelector(seed: 42, windowSize: 5, prime: 0);

        for ($gen = 1; $gen <= 50; $gen++) {
            $wa = $a->forGeneration($gen, 20);
            $wb = $b->forGeneration($gen, 20);
            $this->assertNotNull($wa);
            $this->assertSame(5, $wa->length);
            $this->assertSame(0, $wa->prime);
            // start in [0, rows - window] = [0, 15]
            $this->assertGreaterThanOrEqual(0, $wa->start);
            $this->assertLessThanOrEqual(15, $wa->start);
            // same seed + generation => identical choice (reproducibility)
            $this->assertSame($wa->start, $wb->start);
        }
    }

    public function testWindowStartsAfterThePrimingRows(): void
    {
        // prime=3, window=5 over 20 rows: start must be in [3, 15], and prime rows fit.
        $selector = new WindowSelector(seed: 42, windowSize: 5, prime: 3);
        for ($gen = 1; $gen <= 50; $gen++) {
            $w = $selector->forGeneration($gen, 20);
            $this->assertGreaterThanOrEqual(3, $w->start, 'window starts no earlier than prime');
            $this->assertLessThanOrEqual(15, $w->start);
            $this->assertSame(3, $w->prime);
            $this->assertGreaterThanOrEqual(0, $w->primeStart());
        }
    }

    public function testWindowMovesAcrossGenerations(): void
    {
        $selector = new WindowSelector(seed: 7, windowSize: 3, prime: 0);
        $starts = [];
        for ($gen = 1; $gen <= 30; $gen++) {
            $starts[] = $selector->forGeneration($gen, 20)->start;
        }
        // A moving window: it must not pick the same start every generation.
        $this->assertGreaterThan(1, count(array_unique($starts)));
    }

    public function testCountScorableIgnoresResetRows(): void
    {
        $data = [[[1.0], [0.0]], [], [[2.0], [0.0]], [[3.0], [0.0]], []];
        $this->assertSame(3, WindowSelector::countScorable($data));
    }
}
