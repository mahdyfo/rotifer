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
use Rotifer\Runtime\EvolutionConfig;
use Rotifer\Runtime\Fitness\Problem;
use Rotifer\Runtime\Fitness\SerialEvaluator;
use Rotifer\Tests\Support\Identity;

final class SerialEvaluatorTest extends TestCase
{
    public function testAccumulatesFitnessAcrossRows(): void
    {
        $spec = new NetworkSpec(new Shape(1, 1), false, new Identity());
        $organism = new Organism(
            new Genome([Gene::of(NodeType::Input, 0, NodeType::Output, 0, 1.0)]),
            $spec,
        );

        // Fitness = output each row; outputs equal the inputs under identity.
        $problem = new class implements Problem {
            public function name(): string { return 'sum'; }
            public function shape(): Shape { return new Shape(1, 1); }
            public function config(): EvolutionConfig { return EvolutionConfig::default(); }
            public function data(): array { return [[[2.0], [0.0]], [[3.0], [0.0]]]; }
            public function fitness(Organism $o, array $row): float { return $o->outputs()[0]; }
        };

        (new SerialEvaluator())->evaluate([$organism], $problem);
        $this->assertSame(5.0, $organism->fitness());
    }

    public function testEmptyRowResetsMemoryBetweenSequences(): void
    {
        $spec = new NetworkSpec(new Shape(1, 1), true, new Identity());
        // Self-recurrent accumulator: out = running sum of inputs.
        $organism = new Organism(new Genome([
            Gene::of(NodeType::Input, 0, NodeType::Hidden, 0, 1.0),
            Gene::of(NodeType::Hidden, 0, NodeType::Hidden, 0, 1.0),
            Gene::of(NodeType::Hidden, 0, NodeType::Output, 0, 1.0),
        ]), $spec);

        $problem = new class implements Problem {
            /** @var list<float> */
            public array $captured = [];
            public function name(): string { return 'accumulate'; }
            public function shape(): Shape { return new Shape(1, 1); }
            public function config(): EvolutionConfig { return EvolutionConfig::default()->memory(true); }
            public function data(): array { return [[[1.0], [0.0]], [[1.0], [0.0]], [], [[1.0], [0.0]]]; }
            public function fitness(Organism $o, array $row): float { $this->captured[] = $o->outputs()[0]; return 0.0; }
        };

        (new SerialEvaluator())->evaluate([$organism], $problem);
        // steps before reset accumulate (1, 2); after the empty-row reset it starts at 1 again.
        $this->assertSame([1.0, 2.0, 1.0], $problem->captured);
    }
}
