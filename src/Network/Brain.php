<?php

declare(strict_types=1);

namespace Rotifer\Network;

use Rotifer\Genome\Genome;
use Rotifer\Genome\NodeRef;
use Rotifer\Genome\NodeType;
use Rotifer\Network\Activation\Activation;

/**
 * The runnable network compiled from a genome.
 *
 * The genome is pruned and flattened once at construction into packed, integer
 * indexed arrays - every neuron gets a contiguous slot, and each computed slot
 * carries two parallel lists (source slots, weights). So {@see step()} is a tight
 * integer-indexed weighted-sum loop with no string hashing, no NodeRef allocation
 * and no per-step sorting or lookups (the legacy engine keyed every value and edge
 * by a "type:index" string and re-sorted neurons every step).
 *
 * Slot layout is contiguous: inputs [0, I), outputs [I, I+O), then one slot per
 * hidden neuron in ascending index order. Neuron values live on the Brain, so when
 * the spec has memory they persist between steps (recurrence); otherwise the
 * computed slots are zeroed at the start of each step, giving a pure feed-forward
 * pass. Edges are appended in genome order, so the weighted sum is byte-identical
 * to the old associative-array pass.
 */
final class Brain
{
    private Genome $genome;

    private readonly int $inputCount;
    private readonly int $outputCount;
    private readonly bool $hasMemory;
    private readonly Activation $activation;

    /** Compute order: hidden slots ascending, then output slots ascending. @var list<int> */
    private array $order = [];

    /**
     * Index into {@see $flatSrc}/{@see $flatWt} where each node's incoming edges end,
     * parallel to {@see $order}. Edges are stored grouped by compute order, so node i
     * owns the half-open edge range [prev, edgeEnd[i]). @var list<int>
     */
    private array $edgeEnd = [];

    /** All incoming edges' source slots, flattened in compute order. @var list<int> */
    private array $flatSrc = [];

    /** All incoming edges' weights, flattened in compute order. @var list<float> */
    private array $flatWt = [];

    /** Output slots in index order. @var list<int> */
    private array $outputSlots = [];

    /** Current value of every neuron, indexed by slot. @var list<float> */
    private array $values = [];

    /** slot => node key ("type:index"), for {@see activations()}. @var array<int, string> */
    private array $slotKey = [];

    private int $hiddenCount = 0;
    private int $slotCount = 0;

    public function __construct(Genome $genome, private readonly NetworkSpec $spec)
    {
        $this->genome = GenomePruner::prune($genome, $spec->hasMemory, $spec->layers);
        $this->inputCount = $spec->inputs();
        $this->outputCount = $spec->outputs();
        $this->hasMemory = $spec->hasMemory;
        $this->activation = $spec->activation;
        $this->compile();
    }

    /** The pruned genome this brain runs (what reproduction should inherit). */
    public function genome(): Genome
    {
        return $this->genome;
    }

    private function compile(): void
    {
        $inputs = $this->inputCount;
        $outputs = $this->outputCount;

        // Which hidden neurons the (pruned) genome actually uses.
        $hidden = [];
        foreach ($this->genome->genes() as $gene) {
            foreach ([$gene->from, $gene->to] as $ref) {
                if ($ref->type === NodeType::Hidden) {
                    $hidden[$ref->index] = true;
                }
            }
        }
        ksort($hidden);
        $hiddenIndices = array_keys($hidden);
        $this->hiddenCount = count($hiddenIndices);

        // Slot layout: inputs [0, I), outputs [I, I+O), hidden [I+O, ...).
        $hiddenSlot = [];
        $slot = $inputs + $outputs;
        foreach ($hiddenIndices as $index) {
            $hiddenSlot[$index] = $slot++;
        }
        $this->slotCount = $slot;

        // slot => "type:index" key (built once; never touched in the hot loop).
        for ($i = 0; $i < $inputs; $i++) {
            $this->slotKey[$i] = NodeRef::input($i)->key();
        }
        for ($o = 0; $o < $outputs; $o++) {
            $this->slotKey[$inputs + $o] = NodeRef::output($o)->key();
            $this->outputSlots[] = $inputs + $o;
        }
        foreach ($hiddenIndices as $index) {
            $this->slotKey[$hiddenSlot[$index]] = NodeRef::hidden($index)->key();
        }

        // Compute order: hidden ascending (so chains resolve), then outputs ascending.
        foreach ($hiddenIndices as $index) {
            $this->order[] = $hiddenSlot[$index];
        }
        foreach ($this->outputSlots as $outputSlot) {
            $this->order[] = $outputSlot;
        }

        // Group each gene under its target slot, preserving genome order so the
        // weighted sum stays byte-identical to the legacy associative-array pass.
        $bySource = [];
        $byWeight = [];
        foreach ($this->order as $targetSlot) {
            $bySource[$targetSlot] = [];
            $byWeight[$targetSlot] = [];
        }
        foreach ($this->genome->genes() as $gene) {
            $target = $this->slotFor($gene->to, $hiddenSlot);
            $bySource[$target][] = $this->slotFor($gene->from, $hiddenSlot);
            $byWeight[$target][] = $gene->weight;
        }

        // Flatten into contiguous edge arrays laid out in compute order, with a
        // boundary per node - so the hot loop walks packed lists, no per-node lookup.
        $edge = 0;
        foreach ($this->order as $targetSlot) {
            foreach ($bySource[$targetSlot] as $j => $sourceSlot) {
                $this->flatSrc[$edge] = $sourceSlot;
                $this->flatWt[$edge] = $byWeight[$targetSlot][$j];
                $edge++;
            }
            $this->edgeEnd[] = $edge;
        }

        $this->values = $this->slotCount > 0 ? array_fill(0, $this->slotCount, 0.0) : [];
    }

    /** @param array<int, int> $hiddenSlot hidden index => slot */
    private function slotFor(NodeRef $ref, array $hiddenSlot): int
    {
        return match ($ref->type) {
            NodeType::Input => $ref->index,
            NodeType::Output => $this->inputCount + $ref->index,
            NodeType::Hidden => $hiddenSlot[$ref->index],
        };
    }

    /**
     * Feed one input vector through the network and return the output vector.
     * @param list<float> $inputs
     * @return list<float>
     */
    public function step(array $inputs): array
    {
        $values = &$this->values;
        $inputCount = $this->inputCount;
        for ($i = 0; $i < $inputCount; $i++) {
            $values[$i] = (float) ($inputs[$i] ?? 0.0);
        }

        $order = $this->order;
        if (!$this->hasMemory) {
            foreach ($order as $slot) {
                $values[$slot] = 0.0;
            }
        }

        $activation = $this->activation;
        $flatSrc = $this->flatSrc;
        $flatWt = $this->flatWt;
        $edgeEnd = $this->edgeEnd;
        $edge = 0;
        foreach ($order as $i => $slot) {
            $end = $edgeEnd[$i];
            $sum = 0.0;
            while ($edge < $end) {
                $sum += $values[$flatSrc[$edge]] * $flatWt[$edge];
                $edge++;
            }
            $values[$slot] = $activation->activate($sum);
        }

        return $this->outputs();
    }

    /** @return list<float> output values in index order */
    public function outputs(): array
    {
        $out = [];
        foreach ($this->outputSlots as $slot) {
            $out[] = $this->values[$slot];
        }
        return $out;
    }

    /**
     * Every neuron's current value, keyed by node ("type:index"), as left by the
     * last {@see step()}. Lets the dashboard colour neurons by how strongly they fired.
     *
     * @return array<string, float>
     */
    public function activations(): array
    {
        $out = [];
        foreach ($this->slotKey as $slot => $key) {
            $out[$key] = $this->values[$slot];
        }
        return $out;
    }

    /** Clear all neuron state (between independent sequences). */
    public function resetMemory(): void
    {
        $this->values = $this->slotCount > 0 ? array_fill(0, $this->slotCount, 0.0) : [];
    }

    public function hiddenCount(): int
    {
        return $this->hiddenCount;
    }
}
