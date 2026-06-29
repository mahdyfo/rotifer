<?php

declare(strict_types=1);

namespace Rotifer\Runtime\Fitness;

/**
 * An evaluator that holds resources (e.g. worker processes) needing an orderly
 * shutdown. The World closes it in a finally once the run ends, while the event
 * loop is still alive.
 */
interface ClosableEvaluator extends FitnessEvaluator
{
    public function close(): void;
}
