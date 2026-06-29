<?php

declare(strict_types=1);

namespace Rotifer\Tests\Integration;

use PHPUnit\Framework\TestCase;
use Rotifer\Evolution\OrganismFactory;
use Rotifer\Evolution\World;
use Rotifer\Genome\Gene;
use Rotifer\Genome\NodeType;
use Rotifer\Network\Activation\Sigmoid;
use Rotifer\Network\LayerPlan;
use Rotifer\Network\NetworkSpec;
use Rotifer\Network\Shape;
use Rotifer\Problems\AutoEncoderProblem;
use Rotifer\Runtime\Rng;

/**
 * A run pinned to a fixed layered topology must stay a classic MLP: the neuron
 * count never grows, and no edge ever shortcuts or skips a layer.
 */
final class LayeredTopologyTest extends TestCase
{
    public function testSeedingProducesAFullyConnectedLayeredMlp(): void
    {
        $spec = new NetworkSpec(new Shape(8, 8), false, new Sigmoid(), new LayerPlan([3]));
        $factory = new OrganismFactory($spec, new Rng(1), initialHidden: 0);

        $genome = $factory->random()->genome();

        // 8*3 input->hidden plus 3*8 hidden->output, and nothing else.
        $this->assertCount(8 * 3 + 3 * 8, $genome->genes());
        $this->assertSame(3, $factory->random()->hiddenCount());
        $this->assertLayered($genome->genes(), new LayerPlan([3]));
    }

    public function testEvolvedChampionStaysLayeredWithNoShortcuts(): void
    {
        $problem = new AutoEncoderProblem();
        $config = $problem->config()->generations(5)->population(40)->islands(1)->seed(7);
        $plan = new LayerPlan($config->getHiddenLayers());

        $best = (new World($problem, config: $config))->run();

        // The hidden count never exceeds the planned total (the neuron count is frozen)...
        $this->assertLessThanOrEqual($plan->totalHidden(), $best->hiddenCount());
        $this->assertGreaterThan(0, $best->genome()->count());
        // ...and every surviving edge is a legal one-layer-forward connection, so
        // there are never any direct input->output shortcuts.
        $this->assertLayered($best->genome()->genes(), $plan);
    }

    /** @param list<Gene> $genes */
    private function assertLayered(array $genes, LayerPlan $plan): void
    {
        foreach ($genes as $gene) {
            $this->assertNotTrue(
                $gene->from->type === NodeType::Input && $gene->to->type === NodeType::Output,
                'a layered network must have no direct input->output edge',
            );
            $this->assertTrue(
                $plan->allows($gene->from->type, $gene->from->index, $gene->to->type, $gene->to->index),
                "edge {$gene->connectionKey()} steps exactly one layer forward",
            );
        }
    }
}
