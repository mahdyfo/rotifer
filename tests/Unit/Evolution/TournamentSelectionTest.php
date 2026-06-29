<?php

declare(strict_types=1);

namespace Rotifer\Tests\Unit\Evolution;

use PHPUnit\Framework\TestCase;
use Rotifer\Evolution\Selection\TournamentSelection;
use Rotifer\Genome\Gene;
use Rotifer\Genome\Genome;
use Rotifer\Genome\NodeType;
use Rotifer\Network\Activation\Sigmoid;
use Rotifer\Network\NetworkSpec;
use Rotifer\Network\Shape;
use Rotifer\Organism\Organism;
use Rotifer\Runtime\Rng;

final class TournamentSelectionTest extends TestCase
{
    /** @return list<Organism> */
    private function population(): array
    {
        $spec = new NetworkSpec(new Shape(1, 1), false, new Sigmoid());
        $genome = new Genome([Gene::of(NodeType::Input, 0, NodeType::Output, 0, 1.0)]);
        $organisms = [];
        foreach (range(0, 9) as $i) {
            $organisms[] = (new Organism($genome, $spec, (string) $i))->setFitness((float) $i);
        }
        return $organisms;
    }

    public function testLargeTournamentTendsToPickStrongOrganisms(): void
    {
        $selection = new TournamentSelection(tournamentSize: 8);
        $rng = new Rng(1);
        $sum = 0.0;
        for ($i = 0; $i < 200; $i++) {
            $sum += $selection->pick($this->population(), $rng)->fitness();
        }
        // With 8-way tournaments out of fitnesses 0..9, the mean winner is high.
        $this->assertGreaterThan(7.0, $sum / 200);
    }

    public function testTournamentOfOneIsPlainRandom(): void
    {
        $selection = new TournamentSelection(tournamentSize: 1);
        $picked = $selection->pick($this->population(), new Rng(5));
        $this->assertInstanceOf(Organism::class, $picked);
    }
}
