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
use Rotifer\Runtime\Fitness\Scorer;
use Rotifer\Runtime\Fitness\ScoringWindow;
use Rotifer\Tests\Support\Identity;

final class ScorerWindowTest extends TestCase
{
    public function testScoresOnlyTheWindowRows(): void
    {
        // Identity net (out = in); a capturing problem records which rows it scored.
        $spec = new NetworkSpec(new Shape(1, 1), false, new Identity());
        $organism = new Organism(new Genome([Gene::of(NodeType::Input, 0, NodeType::Output, 0, 1.0)]), $spec);
        $problem = $this->capturing([[[10.0], [0.0]], [[11.0], [0.0]], [[12.0], [0.0]], [[13.0], [0.0]], [[14.0], [0.0]], [[15.0], [0.0]]]);

        // window: score indices 2,3,4 (values 12,13,14); no priming (cold before it).
        $total = Scorer::score($organism, $problem, new ScoringWindow(start: 2, length: 3, prime: 0));

        $this->assertSame([12.0, 13.0, 14.0], $problem->captured);
        $this->assertSame(39.0, $total);
    }

    public function testPrimeFeedsEarlierRowsIntoMemory(): void
    {
        // Recurrent accumulator (out = running sum of inputs), all inputs = 1.
        $problem = $this->capturing([[[1.0], [0.0]], [[1.0], [0.0]], [[1.0], [0.0]], [[1.0], [0.0]], [[1.0], [0.0]]], memory: true);

        // prime=2: rows 0,1 feed the sum to 2 (unscored), then scored rows 2,3 output 3, 4.
        $warm = Scorer::score($this->accumulator(), $problem, new ScoringWindow(start: 2, length: 2, prime: 2));
        $this->assertSame([3.0, 4.0], $problem->captured);
        $this->assertSame(7.0, $warm);
    }

    public function testZeroPrimeStartsColdAtTheWindow(): void
    {
        $problem = $this->capturing([[[1.0], [0.0]], [[1.0], [0.0]], [[1.0], [0.0]], [[1.0], [0.0]], [[1.0], [0.0]]], memory: true);

        $cold = Scorer::score($this->accumulator(), $problem, new ScoringWindow(start: 2, length: 2, prime: 0));
        // rows 0,1 are skipped entirely, so the accumulator starts fresh: 1 then 2.
        $this->assertSame([1.0, 2.0], $problem->captured);
        $this->assertSame(3.0, $cold);
    }

    public function testPartialPrimeFeedsOnlyThePrimingRows(): void
    {
        $problem = $this->capturing([[[1.0], [0.0]], [[1.0], [0.0]], [[1.0], [0.0]], [[1.0], [0.0]], [[1.0], [0.0]]], memory: true);

        // start=3, prime=1: rows 0,1 skipped, row 2 primes the sum to 1, scored rows 3,4 output 2, 3.
        $total = Scorer::score($this->accumulator(), $problem, new ScoringWindow(start: 3, length: 2, prime: 1));
        $this->assertSame([2.0, 3.0], $problem->captured);
        $this->assertSame(5.0, $total);
    }

    public function testNullWindowScoresEveryRow(): void
    {
        $spec = new NetworkSpec(new Shape(1, 1), false, new Identity());
        $organism = new Organism(new Genome([Gene::of(NodeType::Input, 0, NodeType::Output, 0, 1.0)]), $spec);
        $problem = $this->capturing([[[2.0], [0.0]], [[3.0], [0.0]]]);

        $this->assertSame(5.0, Scorer::score($organism, $problem, null));
        $this->assertSame([2.0, 3.0], $problem->captured);
    }

    private function accumulator(): Organism
    {
        $spec = new NetworkSpec(new Shape(1, 1), true, new Identity());
        return new Organism(new Genome([
            Gene::of(NodeType::Input, 0, NodeType::Hidden, 0, 1.0),
            Gene::of(NodeType::Hidden, 0, NodeType::Hidden, 0, 1.0),
            Gene::of(NodeType::Hidden, 0, NodeType::Output, 0, 1.0),
        ]), $spec);
    }

    /** @param list<array{0: list<float>, 1: list<float>}|array{}> $data */
    private function capturing(array $data, bool $memory = false): Problem
    {
        return new class ($data, $memory) implements Problem {
            /** @var list<float> */
            public array $captured = [];
            /** @param list<array{0: list<float>, 1: list<float>}|array{}> $data */
            public function __construct(private array $rows, private bool $memory) {}
            public function name(): string { return 'capture'; }
            public function shape(): Shape { return new Shape(1, 1); }
            public function config(): EvolutionConfig { return EvolutionConfig::default()->memory($this->memory); }
            public function data(): array { return $this->rows; }
            public function fitness(Organism $o, array $row): float { $this->captured[] = $o->outputs()[0]; return $o->outputs()[0]; }
        };
    }
}
