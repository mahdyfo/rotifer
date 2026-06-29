<?php

declare(strict_types=1);

namespace Rotifer\Web;

/**
 * Validates and stores user-authored problems as JSON under problems/custom/.
 * Input comes from HTTP, so it is checked hard: name slugged, widths and row
 * shapes must agree, every value numeric. {@see \Rotifer\Runtime\Fitness\CustomProblem}
 * reads these back.
 */
final class CustomProblemStore
{
    private const MAX_IO = 64;
    private const MAX_ROWS = 500;
    private const MAX_DESCRIPTION = 200;

    private readonly string $dir;

    public function __construct(string $projectRoot)
    {
        $this->dir = rtrim($projectRoot, '/\\') . '/problems/custom';
    }

    /**
     * Validate and persist a definition coming from the UI.
     *
     * @param array<string, mixed> $payload
     * @return array{ok: bool, name?: string, error?: string}
     */
    public function save(array $payload): array
    {
        $label = trim((string) ($payload['name'] ?? ''));
        $slug = $this->slug($label);
        if ($slug === '') {
            return ['ok' => false, 'error' => 'a name is required'];
        }
        $name = 'custom_' . $slug;

        $inputs = (int) ($payload['inputs'] ?? 0);
        $outputs = (int) ($payload['outputs'] ?? 0);
        if ($inputs < 1 || $outputs < 1 || $inputs > self::MAX_IO || $outputs > self::MAX_IO) {
            return ['ok' => false, 'error' => 'inputs and outputs must each be between 1 and ' . self::MAX_IO];
        }

        $rawRows = $payload['rows'] ?? [];
        if (!is_array($rawRows) || $rawRows === []) {
            return ['ok' => false, 'error' => 'at least one example row is required'];
        }
        if (count($rawRows) > self::MAX_ROWS) {
            return ['ok' => false, 'error' => 'too many rows (max ' . self::MAX_ROWS . ')'];
        }

        $rows = [];
        foreach ($rawRows as $i => $row) {
            $input = $this->numbers($row['input'] ?? null);
            $output = $this->numbers($row['output'] ?? null);
            if ($input === null || count($input) !== $inputs) {
                return ['ok' => false, 'error' => "row {$i}: needs {$inputs} numeric input(s)"];
            }
            if ($output === null || count($output) !== $outputs) {
                return ['ok' => false, 'error' => "row {$i}: needs {$outputs} numeric output(s)"];
            }
            $rows[] = ['input' => $input, 'output' => $output];
        }

        $definition = [
            'name' => $name,
            'label' => $label,
            'inputs' => $inputs,
            'outputs' => $outputs,
            'memory' => (bool) ($payload['memory'] ?? false),
            'rows' => $rows,
        ];
        $description = trim((string) ($payload['description'] ?? ''));
        if ($description !== '') {
            $definition['description'] = mb_substr($description, 0, self::MAX_DESCRIPTION);
        }
        foreach (['population', 'generations', 'islands', 'seed'] as $key) {
            if (isset($payload[$key]) && is_numeric($payload[$key])) {
                $definition[$key] = (int) $payload[$key];
            }
        }
        foreach (['trauma', 'adaptive-mutation', 'lifetime-learning'] as $key) {
            if (array_key_exists($key, $payload)) {
                $definition[$key] = (bool) $payload[$key];
            }
        }

        if (!is_dir($this->dir) && !mkdir($this->dir, 0777, true) && !is_dir($this->dir)) {
            return ['ok' => false, 'error' => 'could not create problems/custom'];
        }
        file_put_contents($this->path($name), json_encode($definition, JSON_PRETTY_PRINT | JSON_THROW_ON_ERROR));

        return ['ok' => true, 'name' => $name];
    }

    public function delete(string $name): bool
    {
        $path = $this->path($name);
        if (is_file($path)) {
            return unlink($path);
        }
        return false;
    }

    /** @return array<string, mixed>|null the stored definition for a custom problem name */
    public function definition(string $name): ?array
    {
        $path = $this->path($name);
        if (!is_file($path)) {
            return null;
        }
        $data = json_decode((string) file_get_contents($path), true);
        return is_array($data) ? $data : null;
    }

    /** @return list<array<string, mixed>> every stored definition */
    public function all(): array
    {
        $out = [];
        foreach (glob($this->dir . '/*.json') ?: [] as $file) {
            $data = json_decode((string) file_get_contents($file), true);
            if (is_array($data) && isset($data['name'])) {
                $out[] = $data;
            }
        }
        usort($out, static fn ($a, $b) => ($a['name'] ?? '') <=> ($b['name'] ?? ''));
        return $out;
    }

    private function path(string $name): string
    {
        return $this->dir . '/' . $this->slug($name) . '.json';
    }

    /** @param mixed $values @return list<float>|null */
    private function numbers(mixed $values): ?array
    {
        if (!is_array($values)) {
            return null;
        }
        $out = [];
        foreach ($values as $value) {
            if (!is_numeric($value)) {
                return null;
            }
            $out[] = (float) $value;
        }
        return $out;
    }

    private function slug(string $name): string
    {
        $slug = strtolower(preg_replace('/[^A-Za-z0-9]+/', '_', $name) ?? '');
        return trim($slug, '_');
    }
}
