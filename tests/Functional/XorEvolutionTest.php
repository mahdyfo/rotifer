<?php

declare(strict_types=1);

namespace Rotifer\Tests\Functional;

use PHPUnit\Framework\TestCase;
use Rotifer\Evolution\World;
use Rotifer\Problems\XorProblem;

final class XorEvolutionTest extends TestCase
{
    public function testEvolvesANetworkThatSolvesXor(): void
    {
        $problem = new XorProblem();
        $world = new World($problem);
        $best = $world->run();

        // Re-run the champion over the truth table and check it classifies correctly.
        $best->reset();
        $correct = 0;
        foreach ($problem->data() as $row) {
            $best->step($row[0]);
            $predicted = (int) round($best->outputs()[0]);
            if ($predicted === (int) $row[1][0]) {
                $correct++;
            }
        }

        $this->assertSame(4, $correct, 'champion classifies all four XOR cases');
        $this->assertGreaterThan(3.5, $world->bestFitness());
    }
}
