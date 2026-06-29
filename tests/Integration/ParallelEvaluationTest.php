<?php

declare(strict_types=1);

namespace Rotifer\Tests\Integration;

use PHPUnit\Framework\TestCase;
use Rotifer\Evolution\OrganismFactory;
use Rotifer\Network\LayerPlan;
use Rotifer\Network\NetworkSpec;
use Rotifer\Organism\Organism;
use Rotifer\Problems\AutoEncoderProblem;
use Rotifer\Problems\XorProblem;
use Rotifer\Runtime\Fitness\Problem;
use Rotifer\Runtime\Fitness\SerialEvaluator;
use Rotifer\Runtime\Parallel\ProcessPoolEvaluator;
use Rotifer\Runtime\Rng;

final class ParallelEvaluationTest extends TestCase
{
    /** @return list<Organism> */
    private function buildPopulation(NetworkSpec $spec, int $count): array
    {
        $factory = new OrganismFactory($spec, new Rng(99), initialHidden: 2);
        $organisms = [];
        for ($i = 0; $i < $count; $i++) {
            $organisms[] = $factory->random();
        }
        return $organisms;
    }

    private function specFor(Problem $problem): NetworkSpec
    {
        $config = $problem->config();
        return new NetworkSpec($problem->shape(), $config->hasMemory(), $config->getActivation());
    }

    public function testParallelEvaluationMatchesSerialBitForBit(): void
    {
        $problem = new XorProblem();
        $spec = $this->specFor($problem);

        // Same genomes, two independent populations.
        $template = $this->buildPopulation($spec, 24);
        $serial = array_map(fn (Organism $o) => new Organism($o->genome(), $spec), $template);
        $parallel = array_map(fn (Organism $o) => new Organism($o->genome(), $spec), $template);

        (new SerialEvaluator())->evaluate($serial, $problem);

        $evaluator = new ProcessPoolEvaluator(workers: 4);
        try {
            $evaluator->evaluate($parallel, $problem);
        } finally {
            $evaluator->close();
        }

        foreach ($serial as $i => $organism) {
            $this->assertEqualsWithDelta(
                $organism->fitness(),
                $parallel[$i]->fitness(),
                1e-12,
                "organism #$i scores identically in both evaluators",
            );
        }
    }

    public function testParallelHonoursAnOverriddenSpec(): void
    {
        // Spec pinned to a single hidden layer, different from the problem's own
        // config (which is multi-layer) - the worker must use this spec, not rebuild
        // a stale one from the problem, or the scores diverge
        $problem = new AutoEncoderProblem();
        $config = $problem->config();
        $spec = new NetworkSpec($problem->shape(), $config->hasMemory(), $config->getActivation(), new LayerPlan([4]));

        $template = $this->buildPopulation($spec, 24);
        $serial = array_map(fn (Organism $o) => new Organism($o->genome(), $spec), $template);
        $parallel = array_map(fn (Organism $o) => new Organism($o->genome(), $spec), $template);

        (new SerialEvaluator())->evaluate($serial, $problem);

        $evaluator = new ProcessPoolEvaluator(workers: 4);
        try {
            $evaluator->evaluate($parallel, $problem);
        } finally {
            $evaluator->close();
        }

        foreach ($serial as $i => $organism) {
            $this->assertEqualsWithDelta($organism->fitness(), $parallel[$i]->fitness(), 1e-12, "organism #$i matches under the overridden spec");
        }
    }

    public function testSmallPopulationsFallBackInProcess(): void
    {
        $problem = new XorProblem();
        $spec = $this->specFor($problem);
        $organisms = $this->buildPopulation($spec, 3); // below the parallel threshold

        $evaluator = new ProcessPoolEvaluator(workers: 4);
        try {
            $evaluator->evaluate($organisms, $problem);
        } finally {
            $evaluator->close();
        }

        foreach ($organisms as $organism) {
            $this->assertGreaterThan(0.0, $organism->fitness());
        }
    }
}
