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
use Rotifer\Runtime\Fitness\Predictor;
use Rotifer\Runtime\Fitness\Problem;
use Rotifer\Tests\Support\Identity;

final class PredictorTest extends TestCase
{
    public function testAddsMatchColumnAndOverallSuccessRate(): void
    {
        // A network that passes the single input straight to the output (weight 1).
        $problem = new class implements Problem {
            public function name(): string
            {
                return 'echo';
            }
            public function shape(): Shape
            {
                return new Shape(1, 1);
            }
            public function config(): EvolutionConfig
            {
                return EvolutionConfig::default();
            }
            public function data(): array
            {
                return [
                    [[1.0], [1.0]], // exact -> 100%
                    [[0.5], [1.0]], // off by 0.5 -> 50%
                ];
            }
            public function fitness(Organism $organism, array $row): float
            {
                return 0.0;
            }
        };

        $best = $this->echoOrganism();
        $table = Predictor::describe($problem, $best);

        $this->assertSame(['#', 'input', 'expected', 'predicted', 'match'], $table['columns']);
        $this->assertSame('100%', $table['rows'][0][4]);
        $this->assertSame('50%', $table['rows'][1][4]);
        $this->assertEqualsWithDelta(0.75, $table['successRate'], 1e-9);
    }

    private function echoOrganism(): Organism
    {
        $genome = new Genome([Gene::of(NodeType::Input, 0, NodeType::Output, 0, 1.0)]);
        $spec = new NetworkSpec(new Shape(1, 1), false, new Identity());
        return new Organism($genome, $spec, '0');
    }
}
