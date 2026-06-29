<?php

declare(strict_types=1);

namespace Rotifer\Tests\Unit\Evolution;

use PHPUnit\Framework\TestCase;
use Rotifer\Evolution\Epigenetics\TraumaPolicy;
use Rotifer\Organism\Epigenome;
use Rotifer\Tests\Support\Make;

final class TraumaPolicyTest extends TestCase
{
    public function testStressMarksOnlyBelowAveragePerformers(): void
    {
        $policy = new TraumaPolicy(intensity: 0.5, decay: 0.5);
        $weak = Make::organism(fitness: 0.0, id: 'a');
        $mid = Make::organism(fitness: 1.0, id: 'b');
        $strong = Make::organism(fitness: 2.0, id: 'c'); // average is 1.0

        $policy->applyStress([$weak, $mid, $strong]);

        $this->assertSame(0.5, $weak->epigenome()->intensity('stress'));
        $this->assertSame(0.0, $mid->epigenome()->intensity('stress'));
        $this->assertSame(0.0, $strong->epigenome()->intensity('stress'));
    }

    public function testChildInheritsDecayedStress(): void
    {
        $policy = new TraumaPolicy(intensity: 0.5, decay: 0.5);
        $a = Make::organism(id: 'a', epigenome: new Epigenome(['stress' => 0.8]));
        $b = Make::organism(id: 'b', epigenome: new Epigenome(['stress' => 0.4]));

        $child = $policy->childEpigenome($a, $b);
        // inherit averages to 0.6, then decays by 0.5 -> 0.3
        $this->assertEqualsWithDelta(0.3, $child->intensity('stress'), 1e-9);
    }

    public function testMutationBoostScalesWithStress(): void
    {
        $policy = new TraumaPolicy(intensity: 0.5, decay: 0.5);
        $this->assertSame(1.0, $policy->mutationBoost(new Epigenome()));
        $this->assertEqualsWithDelta(1.7, $policy->mutationBoost(new Epigenome(['stress' => 0.7])), 1e-9);
    }
}
