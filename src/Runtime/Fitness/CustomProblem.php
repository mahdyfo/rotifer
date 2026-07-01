<?php

declare(strict_types=1);

namespace Rotifer\Runtime\Fitness;

use Rotifer\Network\Activation\Sigmoid;
use Rotifer\Network\Shape;
use Rotifer\Organism\Organism;
use Rotifer\Runtime\EvolutionConfig;

/**
 * A {@see Problem} built from a stored definition (name, I/O width, rows, tuning)
 * rather than a hand-written class - what the dashboard's "New problem" form saves.
 * Lives in src/ (not problems/) because it is not zero-arg constructible.
 */
final class CustomProblem implements Problem, Describable
{
    /** @param array<string, mixed> $definition */
    public function __construct(private readonly array $definition)
    {
    }

    public function name(): string
    {
        return (string) ($this->definition['name'] ?? 'custom');
    }

    public function description(): string
    {
        $stored = trim((string) ($this->definition['description'] ?? ''));
        if ($stored !== '') {
            return $stored;
        }
        // No author-supplied blurb, so summarise the shape it was built from.
        return sprintf(
            'Custom task from %d example row(s): %d input(s) -> %d output(s)%s.',
            $this->rowCount(),
            $this->inputs(),
            $this->outputs(),
            $this->hasMemory() ? ', with sequence memory' : '',
        );
    }

    public function shape(): Shape
    {
        return new Shape($this->inputs(), $this->outputs());
    }

    public function config(): EvolutionConfig
    {
        $rows = $this->rowCount();
        $def = $this->definition;

        // Defaults scaled to the dataset, overridden by anything the user set.
        $config = EvolutionConfig::default()
            ->name($this->name())
            ->population((int) ($def['population'] ?? min(300, max(80, $rows * 15))))
            ->generations((int) ($def['generations'] ?? min(200, max(40, $rows * 12))))
            ->islands((int) ($def['islands'] ?? ($rows >= 8 ? 2 : 1)))
            ->seed((int) ($def['seed'] ?? 1234))
            ->surviveRate(0.5)
            ->elitism(2)
            ->initialHidden(max(2, (int) ceil(($this->inputs() + $this->outputs()) / 2)))
            ->memory($this->hasMemory())
            ->activation(new Sigmoid())
            ->crossover(0.5)
            ->mutation(weight: 0.85, addNeuron: 0.05, addConnection: 0.12, removeNeuron: 0.02, removeConnection: 0.03)
            ->weightMutation(count: 2, adjustmentRange: 0.8, randomizeProbability: 0.1)
            ->diversityInjection(0.05)
            ->adaptiveMutation($this->flag('adaptive-mutation', true))
            ->trauma($this->flag('trauma', true))
            ->migration(everyGenerations: 8, topK: 2);

        if ($this->flag('lifetime-learning', false)) {
            $config = $config->lifetimeLearning(steps: 5, lamarckian: 0.3);
        }

        // A random scoring window only makes sense for a sequence (memory) task.
        $window = (int) ($def['window'] ?? 0);
        if ($this->hasMemory() && $window > 0) {
            $config = $config->randomWindow($window, (int) ($def['window-prime'] ?? 0));
        }

        return $config;
    }

    public function data(): array
    {
        $rows = [];
        foreach ($this->definition['rows'] ?? [] as $row) {
            $input = array_map(static fn ($v) => (float) $v, $row['input'] ?? []);
            $output = array_map(static fn ($v) => (float) $v, $row['output'] ?? []);
            $rows[] = [$input, $output];
        }
        return $rows;
    }

    public function fitness(Organism $organism, array $row): float
    {
        $outputs = $organism->outputs();
        $expected = $row[1];
        if ($expected === []) {
            return 0.0;
        }
        $error = 0.0;
        foreach ($expected as $i => $target) {
            $error += abs($target - ($outputs[$i] ?? 0.0));
        }
        return max(0.0, 1.0 - $error / count($expected));
    }

    private function inputs(): int
    {
        return max(1, (int) ($this->definition['inputs'] ?? 1));
    }

    private function outputs(): int
    {
        return max(1, (int) ($this->definition['outputs'] ?? 1));
    }

    private function hasMemory(): bool
    {
        return (bool) ($this->definition['memory'] ?? false);
    }

    private function rowCount(): int
    {
        return is_array($this->definition['rows'] ?? null) ? count($this->definition['rows']) : 0;
    }

    private function flag(string $key, bool $default): bool
    {
        return array_key_exists($key, $this->definition) ? (bool) $this->definition[$key] : $default;
    }
}
