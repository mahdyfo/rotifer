<?php

declare(strict_types=1);

namespace Rotifer\Persistence;

/**
 * Owns the on-disk layout of a run under runs/<name>/ - the live event stream,
 * the run metadata, and the best genome. No side effects in the constructor;
 * directories are created lazily when something is actually written.
 */
final class SnapshotStore
{
    public function __construct(private readonly string $baseDir = 'runs')
    {
    }

    public function runDir(string $name): string
    {
        return $this->baseDir . '/' . $this->slug($name);
    }

    public function streamPath(string $name): string
    {
        return $this->runDir($name) . '/stream.jsonl';
    }

    public function metaPath(string $name): string
    {
        return $this->runDir($name) . '/meta.json';
    }

    public function bestPath(string $name): string
    {
        return $this->runDir($name) . '/best.json';
    }

    public function predictionsPath(string $name): string
    {
        return $this->runDir($name) . '/predictions.json';
    }

    /** The saved population, so a run can be continued instead of restarted. */
    public function checkpointPath(string $name): string
    {
        return $this->runDir($name) . '/checkpoint.json';
    }

    /** @return list<string> run names that have a stream, newest first */
    public function runs(): array
    {
        $dirs = glob($this->baseDir . '/*/stream.jsonl') ?: [];
        usort($dirs, static fn ($a, $b) => filemtime($b) <=> filemtime($a));
        return array_map(static fn ($p) => basename(dirname($p)), $dirs);
    }

    public function ensure(string $name): void
    {
        $dir = $this->runDir($name);
        if (!is_dir($dir)) {
            mkdir($dir, 0777, true);
        }
    }

    private function slug(string $name): string
    {
        $slug = preg_replace('/[^A-Za-z0-9_-]+/', '_', $name) ?? $name;
        return trim($slug, '_') ?: 'run';
    }
}
