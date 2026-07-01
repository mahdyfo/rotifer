<?php

declare(strict_types=1);

namespace Rotifer\Runtime\Parallel;

use Amp\Cancellation;
use Amp\Parallel\Worker\Task;
use Amp\Sync\Channel;
use Rotifer\Genome\Genome;
use Rotifer\Network\NetworkSpec;
use Rotifer\Organism\Organism;
use Rotifer\Runtime\Fitness\Scorer;
use Rotifer\Runtime\Fitness\ScoringWindow;

/**
 * A self-contained unit of fitness evaluation that runs inside an amphp worker.
 *
 * The Problem's class name, the effective NetworkSpec and a batch of genomes cross
 * the process boundary. The worker rebuilds the Problem by name (the reason Problems
 * are classes) but uses the passed spec verbatim, so any run overrides (topology,
 * activation, memory) are honoured and scores match the serial path exactly.
 *
 * @implements Task<list<float>, never, never>
 */
final class EvaluationTask implements Task
{
    /**
     * @param class-string $problemClass
     * @param list<list<array<string, int|float>>> $genomes each element is one genome's Genome::toArray()
     */
    public function __construct(
        private readonly string $problemClass,
        private readonly NetworkSpec $spec,
        private readonly array $genomes,
        private readonly ?ScoringWindow $window = null,
    ) {
    }

    /** @return list<float> fitness per genome, in input order */
    public function run(Channel $channel, Cancellation $cancellation): array
    {
        /** @var \Rotifer\Runtime\Fitness\Problem $problem */
        $problem = new $this->problemClass();

        $fitness = [];
        foreach ($this->genomes as $genomeArray) {
            $organism = new Organism(Genome::fromArray($genomeArray), $this->spec);
            $fitness[] = Scorer::score($organism, $problem, $this->window);
        }
        return $fitness;
    }
}
