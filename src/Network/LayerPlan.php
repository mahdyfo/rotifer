<?php

declare(strict_types=1);

namespace Rotifer\Network;

use Rotifer\Genome\NodeType;

/**
 * A fixed, classic layered topology (a plain feed-forward MLP): inputs, then one
 * or more hidden layers of given sizes, then outputs - wired only between
 * consecutive layers. It is the opt-in alternative to Rotifer's default dynamic
 * NEAT-style topology, for problems (e.g. an auto-encoder) that want a standard
 * network: a bottleneck with no intra-layer edges and no input->output shortcuts.
 *
 * Hidden neurons are numbered contiguously, layer by layer: with sizes [3, 2] the
 * first hidden layer owns indices 0..2 and the second owns 3..4. Because the Brain
 * evaluates hidden neurons in ascending index order, that numbering is already a
 * valid feed-forward order.
 *
 * "Depth" is the layer position used to decide legal edges: inputs are depth 0,
 * hidden layer k is depth k+1, outputs are the last depth. An edge is legal only
 * when it steps exactly one layer forward (depth d -> depth d+1).
 */
final readonly class LayerPlan
{
    /** @var list<int> hidden layer sizes (input/output sizes come from the Shape) */
    public array $sizes;

    /** @param list<int> $sizes */
    public function __construct(array $sizes)
    {
        $this->sizes = array_values(array_filter(array_map(static fn ($n) => max(0, (int) $n), $sizes), static fn (int $n) => $n > 0));
    }

    public function totalHidden(): int
    {
        return array_sum($this->sizes);
    }

    public function layerCount(): int
    {
        return count($this->sizes);
    }

    /** @return list<int> the hidden node indices that make up layer $layer */
    public function hiddenInLayer(int $layer): array
    {
        $start = 0;
        for ($k = 0; $k < $layer; $k++) {
            $start += $this->sizes[$k];
        }
        $indices = [];
        for ($i = 0; $i < ($this->sizes[$layer] ?? 0); $i++) {
            $indices[] = $start + $i;
        }
        return $indices;
    }

    /** Which hidden layer a hidden index belongs to, or -1 if it is out of range. */
    public function layerOfHidden(int $index): int
    {
        if ($index < 0) {
            return -1;
        }
        $bound = 0;
        foreach ($this->sizes as $layer => $size) {
            $bound += $size;
            if ($index < $bound) {
                return $layer;
            }
        }
        return -1;
    }

    /** Layer position of a node: inputs 0, hidden layer k -> k+1, outputs last. */
    private function depthOf(NodeType $type, int $index): int
    {
        return match ($type) {
            NodeType::Input => 0,
            NodeType::Output => $this->layerCount() + 1,
            NodeType::Hidden => ($layer = $this->layerOfHidden($index)) < 0 ? -1 : $layer + 1,
        };
    }

    /** True when an edge steps exactly one layer forward (the only legal move). */
    public function allows(NodeType $fromType, int $fromIndex, NodeType $toType, int $toIndex): bool
    {
        $from = $this->depthOf($fromType, $fromIndex);
        $to = $this->depthOf($toType, $toIndex);
        return $from >= 0 && $to === $from + 1;
    }
}
