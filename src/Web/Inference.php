<?php

declare(strict_types=1);

namespace Rotifer\Web;

use Rotifer\Network\Activation\ActivationFactory;
use Rotifer\Network\Brain;
use Rotifer\Network\NetworkSpec;
use Rotifer\Network\Shape;
use Rotifer\Persistence\Codec\JsonCodec;

/**
 * Runs a saved champion genome on a sequence of custom inputs (one step for a
 * plain network, several for a memory network), returning the final outputs and
 * every neuron's value.
 */
final class Inference
{
    /**
     * @param list<array{0:int,1:int,2:int,3:int,4:float}> $genes
     * @param list<list<float>> $steps one input vector per step
     * @return array{outputs: list<float>, nodes: array<string, float>}
     */
    public static function evaluate(
        array $genes,
        int $inputs,
        int $outputs,
        string $activation,
        bool $memory,
        array $steps,
    ): array {
        $genome = (new JsonCodec())->decode(json_encode($genes, JSON_THROW_ON_ERROR));
        $spec = new NetworkSpec(
            new Shape($inputs, $outputs),
            $memory,
            ActivationFactory::fromName($activation),
        );

        $brain = new Brain($genome, $spec);
        if ($steps === []) {
            $steps = [array_fill(0, $inputs, 0.0)];
        }
        foreach ($steps as $step) {
            $brain->step(array_map(static fn ($v) => (float) $v, array_values($step)));
        }

        return ['outputs' => $brain->outputs(), 'nodes' => $brain->activations()];
    }
}
