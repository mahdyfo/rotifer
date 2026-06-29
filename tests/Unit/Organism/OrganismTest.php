<?php

declare(strict_types=1);

namespace Rotifer\Tests\Unit\Organism;

use PHPUnit\Framework\TestCase;
use Rotifer\Genome\Gene;
use Rotifer\Genome\Genome;
use Rotifer\Genome\NodeType;
use Rotifer\Network\NetworkSpec;
use Rotifer\Network\Shape;
use Rotifer\Organism\Organism;
use Rotifer\Tests\Support\Identity;

final class OrganismTest extends TestCase
{
    private function organism(Genome $genome, bool $memory = false): Organism
    {
        return new Organism($genome, new NetworkSpec(new Shape(1, 1), $memory, new Identity()));
    }

    private function passthroughGenome(): Genome
    {
        return new Genome([Gene::of(NodeType::Input, 0, NodeType::Output, 0, 2.0)]);
    }

    public function testStepThenReadOutputs(): void
    {
        $organism = $this->organism($this->passthroughGenome());
        $organism->step([4.0]);
        $this->assertSame([8.0], $organism->outputs());
    }

    public function testFitnessAccumulationAndReset(): void
    {
        $organism = $this->organism($this->passthroughGenome());
        $organism->addFitness(2.0)->addFitness(3.0);
        $this->assertSame(5.0, $organism->fitness());

        $organism->reset();
        $this->assertSame(0.0, $organism->fitness());
    }

    public function testRankerOrdersByFitnessThenSimplicityThenId(): void
    {
        $rank = Organism::ranker();
        $simple = $this->organism($this->passthroughGenome())->setFitness(1.0)->withId('9');
        $complex = $this->organism(new Genome([
            Gene::of(NodeType::Input, 0, NodeType::Hidden, 0, 1.0),
            Gene::of(NodeType::Hidden, 0, NodeType::Output, 0, 1.0),
        ]))->setFitness(1.0)->withId('1');

        // higher fitness always wins, regardless of complexity
        $fitterButComplex = $this->organism(new Genome([
            Gene::of(NodeType::Input, 0, NodeType::Hidden, 0, 1.0),
            Gene::of(NodeType::Hidden, 0, NodeType::Output, 0, 1.0),
        ]))->setFitness(2.0)->withId('1');
        $this->assertLessThan(0, $rank($fitterButComplex, $simple));

        // equal fitness: the simpler network ranks first even with the larger id
        $this->assertLessThan(0, $rank($simple, $complex));

        // equal fitness and identical shape: lower id breaks the tie deterministically
        $a = $this->organism($this->passthroughGenome())->setFitness(1.0)->withId('3');
        $b = $this->organism($this->passthroughGenome())->setFitness(1.0)->withId('7');
        $this->assertLessThan(0, $rank($a, $b));
    }

    public function testRankerTreatsNearEqualFitnessAsTieAndPrefersSimpler(): void
    {
        $rank = Organism::ranker(3);
        // 4.0 vs 3.99996 differ below the 3 sig-fig resolution, so the simpler net wins
        $simpleLower = $this->organism($this->passthroughGenome())->setFitness(3.99996)->withId('2');
        $complexTop = $this->organism(new Genome([
            Gene::of(NodeType::Input, 0, NodeType::Hidden, 0, 1.0),
            Gene::of(NodeType::Hidden, 0, NodeType::Output, 0, 1.0),
        ]))->setFitness(4.0)->withId('1');
        $this->assertLessThan(0, $rank($simpleLower, $complexTop));

        // a clearly higher score still wins regardless of size
        $complexBetter = $this->organism(new Genome([
            Gene::of(NodeType::Input, 0, NodeType::Hidden, 0, 1.0),
            Gene::of(NodeType::Hidden, 0, NodeType::Output, 0, 1.0),
        ]))->setFitness(4.5)->withId('1');
        $this->assertLessThan(0, $rank($complexBetter, $simpleLower));
    }

    public function testRankerDisabledRanksByExactFitnessIgnoringComplexity(): void
    {
        $rank = Organism::ranker(0);
        // with simplicity off, the higher raw score wins even if it is more complex
        $simpleLower = $this->organism($this->passthroughGenome())->setFitness(3.99996)->withId('2');
        $complexTop = $this->organism(new Genome([
            Gene::of(NodeType::Input, 0, NodeType::Hidden, 0, 1.0),
            Gene::of(NodeType::Hidden, 0, NodeType::Output, 0, 1.0),
        ]))->setFitness(4.0)->withId('1');
        $this->assertLessThan(0, $rank($complexTop, $simpleLower));
    }

    public function testGenomeReflectsPrunedWiring(): void
    {
        // A stray hidden neuron with no path to output should not survive in the genome.
        $organism = $this->organism(new Genome([
            Gene::of(NodeType::Input, 0, NodeType::Output, 0, 1.0),
            Gene::of(NodeType::Input, 0, NodeType::Hidden, 0, 1.0),
        ]));

        $this->assertSame(['0:0->2:0'], array_map(
            fn (Gene $g) => $g->connectionKey(),
            $organism->genome()->genes(),
        ));
        $this->assertSame(0, $organism->hiddenCount());
    }
}
