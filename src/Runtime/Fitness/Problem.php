<?php

declare(strict_types=1);

namespace Rotifer\Runtime\Fitness;

use Rotifer\Network\Shape;
use Rotifer\Organism\Organism;
use Rotifer\Runtime\EvolutionConfig;

/**
 * Everything the engine needs to evolve a solution to one task, in a single
 * class. Defining a Problem is the entire user-facing surface: data + fitness +
 * config. Because it is addressable by class name it can be reconstructed inside
 * a parallel worker (where closures could not travel).
 */
interface Problem
{
    /** Human/CLI identifier, also the run/autosave name. */
    public function name(): string;

    /** Network input/output width. */
    public function shape(): Shape;

    /** Tuning for this problem (population, mutation rates, biology, seed...). */
    public function config(): EvolutionConfig;

    /**
     * Training rows. Each row is [inputs, expectedOutputs]; an empty row ([])
     * signals a memory reset between sequences for memory-enabled networks.
     *
     * @return list<array{0: list<float>, 1: list<float>}|array{}>
     */
    public function data(): array;

    /**
     * Score one already-stepped organism on one data row. Higher is better.
     * Called once per row per organism per generation.
     *
     * @param array{0: list<float>, 1: list<float>} $row
     */
    public function fitness(Organism $organism, array $row): float;
}
