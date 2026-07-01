<?php

declare(strict_types=1);

namespace Rotifer\Runtime\Parallel;

use Amp\Parallel\Worker\ContextWorkerPool;
use Amp\Parallel\Worker\WorkerPool;
use Rotifer\Runtime\Fitness\ClosableEvaluator;
use Rotifer\Runtime\Fitness\Problem;
use Rotifer\Runtime\Fitness\ScoringWindow;
use Rotifer\Runtime\Fitness\SerialEvaluator;

use function Amp\Future\await;

/**
 * Evaluates a population across a pool of worker processes (amphp/parallel).
 *
 * The population is sharded into one {@see EvaluationTask} per worker, the tasks
 * run concurrently, and the returned fitnesses are written back onto the
 * organisms. The pool is created once and reused across generations.
 *
 * Pure scoring only - this is the hot path. Within-lifetime learning needs an
 * Rng per organism and is handled by the serial evaluator, so a tiny population
 * or a learning run transparently falls back to running here in-process.
 */
final class ProcessPoolEvaluator implements ClosableEvaluator
{
    private readonly WorkerPool $pool;
    private readonly SerialEvaluator $fallback;
    private bool $closed = false;

    public function __construct(
        private readonly int $workers = 4,
        ?WorkerPool $pool = null,
    ) {
        $this->pool = $pool ?? new ContextWorkerPool(max(1, $workers));
        $this->fallback = new SerialEvaluator();
    }

    /**
     * Stop the worker pool. Must be called while the event loop is still alive
     * (the World does this in a finally), NOT from a shutdown handler - amphp's
     * shutdown is itself async and would deadlock during PHP teardown. Idempotent.
     */
    public function close(): void
    {
        if ($this->closed) {
            return;
        }
        $this->closed = true;
        $this->pool->shutdown();
    }

    public function evaluate(array $organisms, Problem $problem, ?ScoringWindow $window = null): void
    {
        $count = count($organisms);
        if ($count === 0) {
            return;
        }
        // Not worth the IPC for a handful of organisms.
        if ($count < $this->workers * 2) {
            $this->fallback->evaluate($organisms, $problem, $window);
            return;
        }

        // Workers rebuild the problem by class name with no arguments. A problem that
        // needs constructor arguments (e.g. a user-authored CustomProblem) can't be
        // reconstructed there, so score it serially instead of crashing the worker.
        if (!self::isZeroArgConstructible($problem)) {
            $this->fallback->evaluate($organisms, $problem, $window);
            return;
        }

        $chunkSize = (int) ceil($count / $this->workers);
        $chunks = array_chunk($organisms, $chunkSize);

        // Ship the organisms' actual spec so workers honour run overrides (topology,
        // activation, memory) instead of rebuilding a stale spec from the problem
        $spec = $organisms[array_key_first($organisms)]->spec();

        $executions = [];
        foreach ($chunks as $i => $chunk) {
            $genomes = array_map(static fn ($o) => $o->genome()->toArray(), $chunk);
            $executions[$i] = $this->pool->submit(new EvaluationTask($problem::class, $spec, $genomes, $window));
        }

        $results = await(array_map(static fn ($e) => $e->getFuture(), $executions));

        foreach ($chunks as $i => $chunk) {
            foreach ($chunk as $j => $organism) {
                $organism->setFitness($results[$i][$j]);
            }
        }
    }

    private static function isZeroArgConstructible(Problem $problem): bool
    {
        $constructor = (new \ReflectionClass($problem))->getConstructor();
        return $constructor === null || $constructor->getNumberOfRequiredParameters() === 0;
    }
}
