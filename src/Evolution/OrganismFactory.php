<?php

declare(strict_types=1);

namespace Rotifer\Evolution;

use Rotifer\Genome\Gene;
use Rotifer\Genome\Genome;
use Rotifer\Genome\NodeType;
use Rotifer\Network\NetworkSpec;
use Rotifer\Organism\Organism;
use Rotifer\Runtime\Rng;

/**
 * Builds fresh, randomly-wired organisms - the primordial soup each island and
 * any diversity injection draws from.
 *
 * A seed organism is a fully-connected single hidden layer (every input -> each
 * hidden -> every output) with random weights, guaranteeing at least one
 * input->output path so nothing is dead on arrival. Topology then evolves from
 * here. With zero initial hidden neurons it wires inputs straight to outputs.
 */
final class OrganismFactory
{
    public function __construct(
        private readonly NetworkSpec $spec,
        private readonly Rng $rng,
        private readonly int $initialHidden,
    ) {
    }

    public function random(): Organism
    {
        return new Organism($this->randomGenome(), $this->spec);
    }

    private function randomGenome(): Genome
    {
        if ($this->spec->isLayered()) {
            return $this->layeredGenome();
        }

        $inputs = $this->spec->inputs();
        $outputs = $this->spec->outputs();
        $genes = [];

        if ($this->initialHidden <= 0) {
            for ($i = 0; $i < $inputs; $i++) {
                for ($o = 0; $o < $outputs; $o++) {
                    $genes[] = Gene::of(NodeType::Input, $i, NodeType::Output, $o, $this->rng->weight());
                }
            }
            return new Genome($genes);
        }

        for ($h = 0; $h < $this->initialHidden; $h++) {
            for ($i = 0; $i < $inputs; $i++) {
                $genes[] = Gene::of(NodeType::Input, $i, NodeType::Hidden, $h, $this->rng->weight());
            }
            for ($o = 0; $o < $outputs; $o++) {
                $genes[] = Gene::of(NodeType::Hidden, $h, NodeType::Output, $o, $this->rng->weight());
            }
        }

        return new Genome($genes);
    }

    /**
     * A classic fully-connected layered MLP from the spec's {@see LayerPlan}: each
     * neuron in a layer wired to every neuron in the next, with random weights and
     * no skip or intra-layer edges. Topology is fixed; only the weights evolve.
     */
    private function layeredGenome(): Genome
    {
        $plan = $this->spec->layers;
        $genes = [];

        // Index lists for each successive layer: inputs, hidden layers, outputs.
        $layers = [$this->seq($this->spec->inputs(), NodeType::Input)];
        for ($k = 0; $k < $plan->layerCount(); $k++) {
            $layers[] = array_map(
                static fn (int $h) => [NodeType::Hidden, $h],
                $plan->hiddenInLayer($k),
            );
        }
        $layers[] = $this->seq($this->spec->outputs(), NodeType::Output);

        for ($d = 0; $d < count($layers) - 1; $d++) {
            foreach ($layers[$d] as [$fromType, $fromIdx]) {
                foreach ($layers[$d + 1] as [$toType, $toIdx]) {
                    $genes[] = Gene::of($fromType, $fromIdx, $toType, $toIdx, $this->rng->weight());
                }
            }
        }

        return new Genome($genes);
    }

    /** @return list<array{0: NodeType, 1: int}> the nodes 0..n-1 of one type */
    private function seq(int $n, NodeType $type): array
    {
        $nodes = [];
        for ($i = 0; $i < $n; $i++) {
            $nodes[] = [$type, $i];
        }
        return $nodes;
    }
}
