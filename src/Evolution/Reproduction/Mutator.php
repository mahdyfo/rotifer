<?php

declare(strict_types=1);

namespace Rotifer\Evolution\Reproduction;

use Rotifer\Genome\Gene;
use Rotifer\Genome\Genome;
use Rotifer\Genome\NodeType;
use Rotifer\Genome\Weight;
use Rotifer\Network\NetworkSpec;
use Rotifer\Runtime\EvolutionConfig;
use Rotifer\Runtime\Rng;

/**
 * The five mutation operators, applied in turn, each gated by its own
 * probability: add neuron, add connection, perturb weights, remove neuron,
 * remove connection. Every operator returns a new genome (the input is never
 * mutated in place).
 *
 * A $rateScale lets a caller dial the whole set up or down at once - that hook
 * is what adaptive mutation and epigenetic trauma drive later.
 */
final class Mutator
{
    public function mutate(
        Genome $genome,
        NetworkSpec $spec,
        EvolutionConfig $config,
        Rng $rng,
        float $rateScale = 1.0,
    ): Genome {
        // A fixed layered network has a frozen neuron count, so structural neuron
        // mutations are skipped; only its weights and (re-)connections evolve.
        $layered = $spec->isLayered();

        if (!$layered && $rng->chance($config->getAddNeuronProbability() * $rateScale)) {
            $genome = $this->addNeuron($genome, $spec, $rng);
        }
        if ($rng->chance($config->getAddConnectionProbability() * $rateScale)) {
            $genome = $this->addConnection($genome, $spec, $rng);
        }
        if ($rng->chance($config->getWeightMutationProbability() * $rateScale)) {
            $genome = $this->changeWeights($genome, $config, $rng);
        }
        if (!$layered && $rng->chance($config->getRemoveNeuronProbability() * $rateScale)) {
            $genome = $this->removeNeuron($genome, $rng);
        }
        if ($rng->chance($config->getRemoveConnectionProbability() * $rateScale)) {
            $genome = $this->removeConnection($genome, $rng);
        }

        return $genome;
    }

    private function addNeuron(Genome $genome, NetworkSpec $spec, Rng $rng): Genome
    {
        if ($spec->inputs() === 0 || $spec->outputs() === 0) {
            return $genome;
        }
        $newIndex = $this->maxHiddenIndex($genome) + 1;
        $input = $rng->intBetween(0, $spec->inputs() - 1);
        $output = $rng->intBetween(0, $spec->outputs() - 1);

        return $genome
            ->with(Gene::of(NodeType::Input, $input, NodeType::Hidden, $newIndex, $rng->weight()))
            ->with(Gene::of(NodeType::Hidden, $newIndex, NodeType::Output, $output, $rng->weight()));
    }

    private function addConnection(Genome $genome, NetworkSpec $spec, Rng $rng): Genome
    {
        $candidates = $this->possibleConnections($genome, $spec);
        $existing = [];
        foreach ($genome->genes() as $gene) {
            $existing[$gene->connectionKey()] = true;
        }
        $candidates = array_values(array_filter(
            $candidates,
            static fn (Gene $g) => !isset($existing[$g->connectionKey()]),
        ));
        if ($candidates === []) {
            return $genome;
        }

        $chosen = $candidates[$rng->intBetween(0, count($candidates) - 1)];
        return $genome->with($chosen->withWeight($rng->weight()));
    }

    private function changeWeights(Genome $genome, EvolutionConfig $config, Rng $rng): Genome
    {
        $genes = $genome->genes();
        if ($genes === []) {
            return $genome;
        }
        for ($i = 0; $i < $config->getWeightMutationCount(); $i++) {
            $gene = $genes[$rng->intBetween(0, count($genes) - 1)];
            if ($rng->chance($config->getWeightRandomizeProbability())) {
                $newWeight = $rng->weight();
            } else {
                $delta = $rng->floatBetween(-1.0, 1.0) * $config->getWeightAdjustmentRange();
                $newWeight = Weight::clamp($gene->weight + $delta);
            }
            $genome = $genome->with($gene->withWeight($newWeight));
        }
        return $genome;
    }

    private function removeNeuron(Genome $genome, Rng $rng): Genome
    {
        $hidden = $this->hiddenIndexes($genome);
        if ($hidden === []) {
            return $genome;
        }
        $victim = $hidden[$rng->intBetween(0, count($hidden) - 1)];

        return $genome->map(static function (Gene $gene) use ($victim): ?Gene {
            $touchesVictim =
                ($gene->from->type === NodeType::Hidden && $gene->from->index === $victim)
                || ($gene->to->type === NodeType::Hidden && $gene->to->index === $victim);
            return $touchesVictim ? null : $gene;
        });
    }

    private function removeConnection(Genome $genome, Rng $rng): Genome
    {
        $genes = $genome->genes();
        if ($genes === []) {
            return $genome;
        }
        $victim = $genes[$rng->intBetween(0, count($genes) - 1)];
        return $genome->without($victim->connectionKey());
    }

    /** @return list<Gene> every structurally legal connection, weight-less */
    private function possibleConnections(Genome $genome, NetworkSpec $spec): array
    {
        if ($spec->isLayered()) {
            return $this->layeredConnections($spec);
        }

        $hidden = $this->hiddenIndexes($genome);
        $out = [];

        for ($i = 0; $i < $spec->inputs(); $i++) {
            foreach ($hidden as $h) {
                $out[] = Gene::of(NodeType::Input, $i, NodeType::Hidden, $h, 0.0);
            }
            for ($o = 0; $o < $spec->outputs(); $o++) {
                $out[] = Gene::of(NodeType::Input, $i, NodeType::Output, $o, 0.0);
            }
        }
        foreach ($hidden as $from) {
            foreach ($hidden as $to) {
                if (!$spec->hasMemory && $from >= $to) {
                    continue; // feed-forward only without memory
                }
                if ($spec->hasMemory || $from !== $to) {
                    $out[] = Gene::of(NodeType::Hidden, $from, NodeType::Hidden, $to, 0.0);
                }
            }
            for ($o = 0; $o < $spec->outputs(); $o++) {
                $out[] = Gene::of(NodeType::Hidden, $from, NodeType::Output, $o, 0.0);
            }
        }
        return $out;
    }

    /**
     * Every legal edge in a fixed layered network: input -> first hidden layer,
     * each hidden layer -> the next, and the last hidden layer -> output (or
     * input -> output when there are no hidden layers). Drawn from the plan, not the
     * genome, so a dropped connection between planned neurons can be re-added.
     *
     * @return list<Gene>
     */
    private function layeredConnections(NetworkSpec $spec): array
    {
        $plan = $spec->layers;
        $out = [];

        // Successive "layers" as index lists: inputs, each hidden layer, outputs.
        $layers = [range(0, $spec->inputs() - 1)];
        for ($k = 0; $k < $plan->layerCount(); $k++) {
            $layers[] = $plan->hiddenInLayer($k);
        }
        $layers[] = range(0, $spec->outputs() - 1);

        $typeOf = static fn (int $depth): NodeType => match (true) {
            $depth === 0 => NodeType::Input,
            $depth === count($layers) - 1 => NodeType::Output,
            default => NodeType::Hidden,
        };

        for ($d = 0; $d < count($layers) - 1; $d++) {
            foreach ($layers[$d] as $fromIdx) {
                foreach ($layers[$d + 1] as $toIdx) {
                    $out[] = Gene::of($typeOf($d), $fromIdx, $typeOf($d + 1), $toIdx, 0.0);
                }
            }
        }
        return $out;
    }

    /** @return list<int> */
    private function hiddenIndexes(Genome $genome): array
    {
        $set = [];
        foreach ($genome->genes() as $gene) {
            if ($gene->from->type === NodeType::Hidden) {
                $set[$gene->from->index] = true;
            }
            if ($gene->to->type === NodeType::Hidden) {
                $set[$gene->to->index] = true;
            }
        }
        ksort($set);
        return array_keys($set);
    }

    private function maxHiddenIndex(Genome $genome): int
    {
        $max = -1;
        foreach ($this->hiddenIndexes($genome) as $index) {
            $max = max($max, $index);
        }
        return $max;
    }
}
