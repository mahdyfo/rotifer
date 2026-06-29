<?php

declare(strict_types=1);

namespace Rotifer\Cli;

use RuntimeException;
use Rotifer\Runtime\Fitness\CustomProblem;
use Rotifer\Runtime\Fitness\Problem;
use Rotifer\Web\CustomProblemStore;

/**
 * Finds Problem classes by friendly name. Accepts a fully-qualified class name,
 * a short class name ("XorProblem"), or a loose alias ("xor", "memory-recall")
 * that is normalized to a class. Also resolves user-authored problems stored as
 * JSON under problems/custom/, and lists everything for the `list` command.
 */
final class ProblemRegistry
{
    private readonly CustomProblemStore $customStore;

    public function __construct(private readonly string $problemsDir, ?CustomProblemStore $customStore = null)
    {
        $this->customStore = $customStore ?? new CustomProblemStore(dirname($problemsDir));
    }

    public function resolve(string $name): Problem
    {
        foreach ($this->candidateClasses($name) as $class) {
            if (class_exists($class) && is_subclass_of($class, Problem::class)) {
                /** @var Problem */
                return new $class();
            }
        }

        $definition = $this->customStore->definition($name);
        if ($definition !== null) {
            return new CustomProblem($definition);
        }

        throw new RuntimeException(
            "Unknown problem \"{$name}\". Run `rotifer list` to see available problems."
        );
    }

    /** @return list<array{name: string, class: string, custom: bool}> */
    public function all(): array
    {
        $found = [];
        foreach (glob($this->problemsDir . '/*.php') ?: [] as $file) {
            $class = 'Rotifer\\Problems\\' . basename($file, '.php');
            if (class_exists($class) && is_subclass_of($class, Problem::class)) {
                /** @var Problem $instance */
                $instance = new $class();
                $found[] = ['name' => $instance->name(), 'class' => $class, 'custom' => false];
            }
        }
        foreach ($this->customStore->all() as $definition) {
            $found[] = ['name' => (string) $definition['name'], 'class' => CustomProblem::class, 'custom' => true];
        }
        usort($found, static fn ($a, $b) => $a['name'] <=> $b['name']);
        return $found;
    }

    /** @return list<string> */
    private function candidateClasses(string $name): array
    {
        $pascal = str_replace(' ', '', ucwords(str_replace(['-', '_'], ' ', $name)));

        return array_values(array_unique([
            $name,
            'Rotifer\\Problems\\' . $name,
            'Rotifer\\Problems\\' . $pascal,
            'Rotifer\\Problems\\' . $pascal . 'Problem',
        ]));
    }
}
