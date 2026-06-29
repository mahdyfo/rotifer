<?php

declare(strict_types=1);

namespace Rotifer\Tests\Unit\Runtime;

use PHPUnit\Framework\TestCase;
use Rotifer\Genome\Gene;
use Rotifer\Genome\Genome;
use Rotifer\Genome\NodeType;
use Rotifer\Network\NetworkSpec;
use Rotifer\Network\Shape;
use Rotifer\Organism\Organism;
use Rotifer\Runtime\Fitness\CustomProblem;
use Rotifer\Tests\Support\Identity;

final class CustomProblemTest extends TestCase
{
    private function definition(array $overrides = []): array
    {
        return array_merge([
            'name' => 'custom_and',
            'inputs' => 2,
            'outputs' => 1,
            'memory' => false,
            'rows' => [
                ['input' => [0, 0], 'output' => [0]],
                ['input' => [0, 1], 'output' => [0]],
                ['input' => [1, 0], 'output' => [0]],
                ['input' => [1, 1], 'output' => [1]],
            ],
        ], $overrides);
    }

    public function testExposesNameShapeAndData(): void
    {
        $problem = new CustomProblem($this->definition());

        $this->assertSame('custom_and', $problem->name());
        $this->assertSame(2, $problem->shape()->inputs);
        $this->assertSame(1, $problem->shape()->outputs);
        $this->assertSame([[1.0, 1.0], [1.0]], $problem->data()[3]);
    }

    public function testRecommendsDefaultsButHonoursOverrides(): void
    {
        $recommended = (new CustomProblem($this->definition()))->config();
        $this->assertGreaterThanOrEqual(80, $recommended->getPopulation());
        $this->assertTrue($recommended->isTraumaEnabled()); // biology on by default

        $tuned = (new CustomProblem($this->definition([
            'population' => 42,
            'generations' => 7,
            'trauma' => false,
        ])))->config();
        $this->assertSame(42, $tuned->getPopulation());
        $this->assertSame(7, $tuned->getGenerations());
        $this->assertFalse($tuned->isTraumaEnabled());
    }

    public function testFitnessRewardsCloseness(): void
    {
        $problem = new CustomProblem($this->definition());
        $organism = $this->constantOrganism();

        // Output is a constant 1.0; expected 1.0 -> perfect, expected 0.0 -> worst.
        $this->assertEqualsWithDelta(1.0, $problem->fitness($organism, [[1, 1], [1.0]]), 1e-9);
        $this->assertEqualsWithDelta(0.0, $problem->fitness($organism, [[0, 0], [0.0]]), 1e-9);
    }

    private function constantOrganism(): Organism
    {
        // bias-style: input 0 is always 1 in the rows above, weight 1 -> output 1.
        $genome = new Genome([Gene::of(NodeType::Input, 0, NodeType::Output, 0, 1.0)]);
        $spec = new NetworkSpec(new Shape(2, 1), false, new Identity());
        $organism = new Organism($genome, $spec, '0');
        $organism->step([1.0, 1.0]);
        return $organism;
    }
}
