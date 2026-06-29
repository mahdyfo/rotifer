<?php

declare(strict_types=1);

namespace Rotifer\Tests\Support;

use Rotifer\Genome\Gene;
use Rotifer\Genome\Genome;
use Rotifer\Genome\NodeType;
use Rotifer\Network\NetworkSpec;
use Rotifer\Network\Shape;
use Rotifer\Organism\Epigenome;
use Rotifer\Organism\Organism;

/** Small factory of trivial organisms for biology-policy unit tests. */
final class Make
{
    public static function spec(int $inputs = 1, int $outputs = 1, bool $memory = false): NetworkSpec
    {
        return new NetworkSpec(new Shape($inputs, $outputs), $memory, new Identity());
    }

    public static function organism(float $fitness = 0.0, string $id = '0', ?Epigenome $epigenome = null): Organism
    {
        $genome = new Genome([Gene::of(NodeType::Input, 0, NodeType::Output, 0, 1.0)]);
        $organism = new Organism($genome, self::spec(), $id, $epigenome);
        $organism->setFitness($fitness);
        return $organism;
    }
}
