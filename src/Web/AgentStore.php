<?php

declare(strict_types=1);

namespace Rotifer\Web;

/**
 * A small library of saved champions ("agents") under saved_agents/*.json.
 *
 * When a run finishes or is stopped, its best organism is saved here so it
 * survives its run/ folder: the genome (a compact hex string) plus everything
 * needed to run it again (I/O width, activation, memory) and a little provenance
 * (problem, fitness, match rate, size). The dashboard lists them, loads one into
 * the Champion-predictions panel, and runs inputs through it via {@see Inference}.
 *
 * Files are written as compact JSON (no pretty-printing) and the genome is stored
 * hex-encoded rather than as a JSON tuple array, so a saved agent stays small.
 */
final class AgentStore
{
    private const MAX_NAME = 60;

    private readonly string $dir;

    public function __construct(string $projectRoot)
    {
        $this->dir = rtrim($projectRoot, '/\\') . '/saved_agents';
    }

    /**
     * Persist an agent under a slug derived from its name (the run name for an
     * auto-save), overwriting any existing slot with that slug. `genome` (a hex
     * string) and a name are required; the rest is provenance.
     *
     * @param array<string, mixed> $agent
     * @return array{ok: bool, name?: string, path?: string, error?: string}
     */
    public function save(array $agent): array
    {
        $label = trim((string) ($agent['name'] ?? ''));
        $slug = $this->slug($label);
        if ($slug === '') {
            return ['ok' => false, 'error' => 'a name is required'];
        }
        $genome = $agent['genome'] ?? null;
        if (!is_string($genome) || $genome === '') {
            return ['ok' => false, 'error' => 'no champion to save yet'];
        }

        $record = [
            'name' => mb_substr($label, 0, self::MAX_NAME),
            'slug' => $slug,
            'problem' => (string) ($agent['problem'] ?? ''),
            'inputs' => max(1, (int) ($agent['inputs'] ?? 1)),
            'outputs' => max(1, (int) ($agent['outputs'] ?? 1)),
            'activation' => (string) ($agent['activation'] ?? 'sigmoid'),
            'memory' => (bool) ($agent['memory'] ?? false),
            'fitness' => (float) ($agent['fitness'] ?? 0.0),
            'matchRate' => isset($agent['matchRate']) ? (float) $agent['matchRate'] : null,
            'hidden' => (int) ($agent['hidden'] ?? 0),
            'geneCount' => (int) ($agent['geneCount'] ?? 0),
            'savedAt' => date('c'),
            'genome' => $genome,
        ];

        if (!is_dir($this->dir) && !mkdir($this->dir, 0777, true) && !is_dir($this->dir)) {
            return ['ok' => false, 'error' => 'could not create saved_agents/'];
        }
        file_put_contents($this->path($slug), json_encode($record, JSON_THROW_ON_ERROR));

        return ['ok' => true, 'name' => $slug, 'path' => $this->relativePath($slug)];
    }

    /**
     * Every saved agent as a lightweight summary (no genome), newest first - what
     * the dropdown lists. Each carries its on-disk `path` for display.
     *
     * @return list<array<string, mixed>>
     */
    public function all(): array
    {
        $out = [];
        foreach (glob($this->dir . '/*.json') ?: [] as $file) {
            $data = json_decode((string) file_get_contents($file), true);
            if (is_array($data) && isset($data['slug'])) {
                unset($data['genome']);
                $data['path'] = $this->relativePath((string) $data['slug']);
                $data['savedAtTs'] = @filemtime($file) ?: 0;
                $out[] = $data;
            }
        }
        usort($out, static fn ($a, $b) => ($b['savedAtTs'] ?? 0) <=> ($a['savedAtTs'] ?? 0));
        return $out;
    }

    /** @return array<string, mixed>|null the full stored agent (including the hex genome). */
    public function get(string $slug): ?array
    {
        $path = $this->path($this->slug($slug));
        if (!is_file($path)) {
            return null;
        }
        $data = json_decode((string) file_get_contents($path), true);
        return is_array($data) ? $data : null;
    }

    public function delete(string $slug): bool
    {
        $path = $this->path($this->slug($slug));
        return is_file($path) && unlink($path);
    }

    private function path(string $slug): string
    {
        return $this->dir . '/' . $slug . '.json';
    }

    private function relativePath(string $slug): string
    {
        return 'saved_agents/' . $slug . '.json';
    }

    private function slug(string $name): string
    {
        $slug = strtolower(preg_replace('/[^A-Za-z0-9]+/', '_', $name) ?? '');
        return trim($slug, '_');
    }
}
