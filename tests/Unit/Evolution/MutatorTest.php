<?php

declare(strict_types=1);

namespace Rotifer\Tests\Unit\Evolution;

use PHPUnit\Framework\TestCase;
use Rotifer\Evolution\Reproduction\Mutator;
use Rotifer\Genome\Gene;
use Rotifer\Genome\Genome;
use Rotifer\Genome\NodeType;
use Rotifer\Network\Activation\Sigmoid;
use Rotifer\Network\NetworkSpec;
use Rotifer\Network\Shape;
use Rotifer\Runtime\EvolutionConfig;
use Rotifer\Runtime\Rng;

final class MutatorTest extends TestCase
{
    private function spec(): NetworkSpec
    {
        return new NetworkSpec(new Shape(2, 1), false, new Sigmoid());
    }

    private function baseGenome(): Genome
    {
        return new Genome([
            Gene::of(NodeType::Input, 0, NodeType::Output, 0, 1.0),
            Gene::of(NodeType::Input, 1, NodeType::Output, 0, 1.0),
        ]);
    }

    public function testNoMutationWhenAllRatesAreZero(): void
    {
        $config = EvolutionConfig::default()->mutation(0.0, 0.0, 0.0, 0.0, 0.0);
        $genome = $this->baseGenome();

        $result = (new Mutator())->mutate($genome, $this->spec(), $config, new Rng(1));
        $this->assertSame($genome->toArray(), $result->toArray());
    }

    public function testAddNeuronIntroducesAHiddenPath(): void
    {
        $config = EvolutionConfig::default()->mutation(0.0, addNeuron: 1.0, addConnection: 0.0, removeNeuron: 0.0, removeConnection: 0.0);

        $result = (new Mutator())->mutate($this->baseGenome(), $this->spec(), $config, new Rng(3));

        $hasHidden = false;
        foreach ($result->genes() as $gene) {
            if ($gene->from->type === NodeType::Hidden || $gene->to->type === NodeType::Hidden) {
                $hasHidden = true;
            }
        }
        $this->assertTrue($hasHidden, 'a hidden neuron and its connections were added');
    }

    public function testChangeWeightStaysWithinLegalRange(): void
    {
        $config = EvolutionConfig::default()
            ->mutation(weight: 1.0, addNeuron: 0.0, addConnection: 0.0, removeNeuron: 0.0, removeConnection: 0.0)
            ->weightMutation(count: 5, adjustmentRange: 100.0, randomizeProbability: 0.5);

        $result = (new Mutator())->mutate($this->baseGenome(), $this->spec(), $config, new Rng(9));
        foreach ($result->genes() as $gene) {
            $this->assertGreaterThanOrEqual(-\Rotifer\Genome\Weight::MAX, $gene->weight);
            $this->assertLessThanOrEqual(\Rotifer\Genome\Weight::MAX, $gene->weight);
        }
    }

    public function testRemoveConnectionShrinksGenome(): void
    {
        $config = EvolutionConfig::default()->mutation(0.0, 0.0, 0.0, 0.0, removeConnection: 1.0);
        $result = (new Mutator())->mutate($this->baseGenome(), $this->spec(), $config, new Rng(2));
        $this->assertLessThan($this->baseGenome()->count(), $result->count());
    }

    public function testMutationDoesNotAlterTheInputGenome(): void
    {
        $config = EvolutionConfig::default()->mutation(weight: 1.0, addNeuron: 1.0, addConnection: 1.0, removeNeuron: 1.0, removeConnection: 1.0);
        $genome = $this->baseGenome();
        $before = $genome->toArray();

        (new Mutator())->mutate($genome, $this->spec(), $config, new Rng(11));
        $this->assertSame($before, $genome->toArray(), 'original genome is immutable');
    }
}
