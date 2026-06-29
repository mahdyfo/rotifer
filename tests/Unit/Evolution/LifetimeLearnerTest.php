<?php

declare(strict_types=1);

namespace Rotifer\Tests\Unit\Evolution;

use PHPUnit\Framework\TestCase;
use Rotifer\Evolution\Learning\LifetimeLearner;
use Rotifer\Genome\Gene;
use Rotifer\Genome\Genome;
use Rotifer\Genome\NodeType;
use Rotifer\Organism\Organism;
use Rotifer\Runtime\EvolutionConfig;
use Rotifer\Runtime\Fitness\Problem;
use Rotifer\Runtime\Rng;
use Rotifer\Network\Shape;
use Rotifer\Tests\Support\Make;

final class LifetimeLearnerTest extends TestCase
{
    /** A problem where fitness == the (identity) output, so bigger weight = better. */
    private function climbProblem(): Problem
    {
        return new class implements Problem {
            public function name(): string { return 'climb'; }
            public function shape(): Shape { return new Shape(1, 1); }
            public function config(): EvolutionConfig { return EvolutionConfig::default(); }
            public function data(): array { return [[[1.0], [0.0]]]; }
            public function fitness(Organism $o, array $row): float { return $o->outputs()[0]; }
        };
    }

    private function organism(): Organism
    {
        return new Organism(
            new Genome([Gene::of(NodeType::Input, 0, NodeType::Output, 0, 0.0)]),
            Make::spec(),
        );
    }

    public function testLearningImprovesFitness(): void
    {
        $organism = $this->organism();
        $learner = new LifetimeLearner(new Rng(1), steps: 60, stepSize: 1.0, lamarckianFraction: 0.0);

        $learner->refine($organism, $this->climbProblem());
        $this->assertGreaterThan(0.5, $organism->fitness(), 'Baldwin: learned phenotype scores better than the 0-weight start');
    }

    public function testPureBaldwinLeavesGenomeUnchanged(): void
    {
        $organism = $this->organism();
        $learner = new LifetimeLearner(new Rng(1), steps: 60, stepSize: 1.0, lamarckianFraction: 0.0);

        $learner->refine($organism, $this->climbProblem());
        $this->assertSame(0.0, $organism->genome()->genes()[0]->weight, 'nothing is written back at fraction 0');
    }

    public function testLamarckianWritesLearningBackIntoTheGenome(): void
    {
        $organism = $this->organism();
        $learner = new LifetimeLearner(new Rng(1), steps: 60, stepSize: 1.0, lamarckianFraction: 1.0);

        $learner->refine($organism, $this->climbProblem());
        $this->assertGreaterThan(0.5, $organism->genome()->genes()[0]->weight, 'the learned weight is inherited at fraction 1');
    }
}
