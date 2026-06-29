<?php

declare(strict_types=1);

namespace Rotifer\Network;

use Rotifer\Genome\Gene;
use Rotifer\Genome\Genome;
use Rotifer\Genome\NodeType;

/**
 * Removes genes that can have no effect, yielding a genome whose every hidden
 * neuron lies on at least one input -> output path. This is the cleaner,
 * single-purpose successor to the legacy Agent::deleteRedundantGenes().
 *
 * Rules:
 *  - structurally illegal edges are dropped (into an input, out of an output,
 *    input -> input);
 *  - without memory the network must be strictly feed-forward by hidden index,
 *    so self-loops and "future -> earlier" hidden edges are dropped (they would
 *    only ever read a stale/zero value);
 *  - hidden neurons that end up with no input or no output are pruned, repeated
 *    to a fixpoint so chains collapse cleanly.
 */
final class GenomePruner
{
    public static function prune(Genome $genome, bool $hasMemory, ?LayerPlan $layers = null): Genome
    {
        $layered = $layers !== null && $layers->layerCount() > 0 ? $layers : null;
        $genes = [];
        foreach ($genome->genes() as $gene) {
            if (self::isLegal($gene, $hasMemory, $layered)) {
                $genes[] = $gene;
            }
        }

        // A fixed layered network keeps every planned neuron; only the dynamic
        // topology prunes hidden neurons that lost their input or output path.
        if ($layered === null) {
            $genes = self::dropStrayHidden($genes);
        }

        return new Genome($genes);
    }

    private static function isLegal(Gene $gene, bool $hasMemory, ?LayerPlan $layers): bool
    {
        $from = $gene->from;
        $to = $gene->to;

        if ($to->type === NodeType::Input) {
            return false; // nothing connects into an input
        }
        if ($from->type === NodeType::Output) {
            return false; // an output never feeds anything
        }
        if ($from->type === NodeType::Input && $to->type === NodeType::Input) {
            return false;
        }

        // A fixed layered network only allows edges one layer forward - no
        // intra-layer, no skip (so no input->output), no backward.
        if ($layers !== null) {
            return $layers->allows($from->type, $from->index, $to->type, $to->index);
        }

        // Without memory, hidden wiring must flow from lower to higher index only.
        if (!$hasMemory
            && $from->type === NodeType::Hidden
            && $to->type === NodeType::Hidden
            && $from->index >= $to->index
        ) {
            return false;
        }

        return true;
    }

    /**
     * @param list<Gene> $genes
     * @return list<Gene>
     */
    private static function dropStrayHidden(array $genes): array
    {
        while (true) {
            $hiddenWithInput = [];
            $hiddenWithOutput = [];
            foreach ($genes as $gene) {
                if ($gene->to->type === NodeType::Hidden) {
                    $hiddenWithInput[$gene->to->index] = true;
                }
                if ($gene->from->type === NodeType::Hidden) {
                    $hiddenWithOutput[$gene->from->index] = true;
                }
            }

            $kept = [];
            $removedAny = false;
            foreach ($genes as $gene) {
                if ($gene->from->type === NodeType::Hidden
                    && !isset($hiddenWithInput[$gene->from->index])) {
                    $removedAny = true;
                    continue;
                }
                if ($gene->to->type === NodeType::Hidden
                    && !isset($hiddenWithOutput[$gene->to->index])) {
                    $removedAny = true;
                    continue;
                }
                $kept[] = $gene;
            }

            $genes = $kept;
            if (!$removedAny) {
                return $genes;
            }
        }
    }
}
