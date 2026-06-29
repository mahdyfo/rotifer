<?php

declare(strict_types=1);

/**
 * Router + control plane for the dashboard (`php -S host:port src/Web/server.php`).
 *
 * One persistent server serves the UI and a small API over the whole runs/
 * directory - not a single run. The browser can list problems and their
 * defaults, start and stop runs, and follow whichever run is active. Every data
 * request is short (the client polls), so the single-worker dev server stays
 * responsive on every platform.
 */

require dirname(__DIR__, 2) . '/vendor/autoload.php';

use Rotifer\Cli\ProblemRegistry;
use Rotifer\Persistence\SnapshotStore;
use Rotifer\Runtime\FastRuntime;
use Rotifer\Web\CustomProblemStore;
use Rotifer\Web\Inference;
use Rotifer\Web\RunManager;

$root = dirname(__DIR__, 2);
$publicDir = __DIR__ . '/public';
$store = new SnapshotStore($root . '/runs');
$customStore = new CustomProblemStore($root);
$registry = new ProblemRegistry($root . '/problems', $customStore);
$manager = new RunManager($root, $store);

$uri = parse_url($_SERVER['REQUEST_URI'] ?? '/', PHP_URL_PATH) ?: '/';
$method = $_SERVER['REQUEST_METHOD'] ?? 'GET';

if (str_starts_with($uri, '/api/')) {
    header('Content-Type: application/json');
    // The client polls fixed URLs (e.g. /api/status); never let the browser serve
    // a cached response or the dashboard shows stale state.
    header('Cache-Control: no-store');
    echo json_encode(handleApi($uri, $method, $registry, $store, $manager, $customStore), JSON_THROW_ON_ERROR);
    return true;
}

$path = $uri === '/' ? '/index.html' : $uri;
$file = realpath($publicDir . $path);
if ($file !== false && str_starts_with($file, (string) realpath($publicDir)) && is_file($file)) {
    header('Content-Type: ' . contentType($file));
    // Never cache the dashboard assets: this is a dev server and a stale app.js
    // would silently keep running the old client (e.g. ignoring Start).
    header('Cache-Control: no-store');
    readfile($file);
    return true;
}

http_response_code(404);
echo 'Not found';
return true;

function handleApi(string $uri, string $method, ProblemRegistry $registry, SnapshotStore $store, RunManager $manager, CustomProblemStore $customStore): mixed
{
    return match ($uri) {
        '/api/problems' => listProblems($registry),
        '/api/problemdata' => problemData($registry, (string) ($_GET['name'] ?? '')),
        '/api/problems/create' => $method === 'POST' ? $customStore->save(requestBody()) : ['ok' => false, 'error' => 'POST required'],
        '/api/problems/delete' => $method === 'POST' ? deleteProblem($customStore, $manager, requestBody()) : ['ok' => false, 'error' => 'POST required'],
        '/api/status' => $manager->status(),
        '/api/health' => runtimeHealth($store, $manager),
        '/api/start' => $method === 'POST' ? startRun($manager, requestBody()) : ['ok' => false, 'error' => 'POST required'],
        '/api/stop' => doStop($manager),
        '/api/stream' => streamRecords($store, runParam($manager), (int) ($_GET['from'] ?? 0)),
        '/api/meta' => readJson($store->metaPath(runParam($manager))),
        '/api/best' => readJson($store->bestPath(runParam($manager))),
        '/api/predictions' => readJson($store->predictionsPath(runParam($manager))),
        '/api/infer' => inferOutput($store, runParam($manager), (string) ($_GET['input'] ?? '')),
        default => ['error' => 'unknown endpoint'],
    };
}

/** @return list<array<string, mixed>> */
function listProblems(ProblemRegistry $registry): array
{
    $out = [];
    foreach ($registry->all() as $entry) {
        $problem = $registry->resolve($entry['name']);
        $c = $problem->config();
        $out[] = [
            'name' => $problem->name(),
            'custom' => $entry['custom'] ?? false,
            'inputs' => $problem->shape()->inputs,
            'outputs' => $problem->shape()->outputs,
            'memory' => $c->hasMemory(),
            // Every settable knob, keyed exactly like the CLI flags and the /api/start
            // overrides (RunOptions is the shared source of truth), plus `parallel`,
            // which the UI shows as a pre-checked box rather than a value field.
            'defaults' => \Rotifer\Runtime\RunOptions::defaultsOf($c) + [
                'parallel' => $c->isParallelEnabled(),
            ],
        ];
    }
    return $out;
}

/** Shape, memory flag and example rows for one problem, so the UI can show/edit them. */
function problemData(ProblemRegistry $registry, string $name): array
{
    if ($name === '') {
        return ['ok' => false, 'error' => 'name required'];
    }
    try {
        $problem = $registry->resolve($name);
    } catch (\Throwable $e) {
        return ['ok' => false, 'error' => $e->getMessage()];
    }

    $rows = [];
    foreach ($problem->data() as $row) {
        if ($row === []) {
            continue;
        }
        $rows[] = ['input' => $row[0], 'output' => $row[1]];
        if (count($rows) >= 200) {
            break;
        }
    }

    $episodic = $problem instanceof \Rotifer\Runtime\Fitness\Predictable && count($rows) <= 1;
    return [
        'ok' => true,
        'name' => $problem->name(),
        'inputs' => $problem->shape()->inputs,
        'outputs' => $problem->shape()->outputs,
        'memory' => $problem->config()->hasMemory(),
        'custom' => str_starts_with($problem->name(), 'custom_'),
        'episodic' => $episodic,
        'rows' => $rows,
    ];
}

function deleteProblem(CustomProblemStore $customStore, RunManager $manager, array $body): array
{
    $name = (string) ($body['name'] ?? '');
    if ($name === '' || !str_starts_with($name, 'custom_')) {
        return ['ok' => false, 'error' => 'only custom problems can be deleted'];
    }
    return ['ok' => $customStore->delete($name)];
}

/**
 * Run the latest champion of a run on one user-supplied input and return both the
 * outputs and every neuron's value, so the dashboard can colour the network.
 */
function inferOutput(SnapshotStore $store, string $run, string $rawInput): array
{
    $record = lastRecord($store, $run);
    $network = $record['network'] ?? null;
    if (!is_array($network) || !isset($network['genes'])) {
        return ['ok' => false, 'error' => 'no champion to run yet'];
    }
    $meta = readJson($store->metaPath($run));
    $activation = is_array($meta) ? (string) ($meta['activation'] ?? 'sigmoid') : 'sigmoid';
    $memory = is_array($meta) ? (bool) ($meta['memory'] ?? false) : false;

    $inputs = (int) $network['inputs'];
    $outputs = (int) $network['outputs'];
    // Steps are separated by ';' (a memory network is fed one step at a time).
    $steps = array_map(static fn (string $s) => parseVector($s, $inputs), explode(';', $rawInput));

    $result = Inference::evaluate($network['genes'], $inputs, $outputs, $activation, $memory, $steps);
    return ['ok' => true, 'steps' => count($steps), 'activation' => $activation, 'outputs' => $result['outputs'], 'nodes' => $result['nodes']];
}

/** @return array<string, mixed>|null the final streamed generation record */
function lastRecord(SnapshotStore $store, string $run): ?array
{
    $path = $store->streamPath($run);
    $lines = is_file($path) ? file($path, FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES) : [];
    for ($i = count($lines) - 1; $i >= 0; $i--) {
        $decoded = json_decode($lines[$i], true);
        if (is_array($decoded)) {
            return $decoded;
        }
    }
    return null;
}

/** @return list<float> a fixed-width input vector parsed from "1, 0, 1" */
function parseVector(string $raw, int $width): array
{
    $parts = $raw === '' ? [] : preg_split('/\s*,\s*/', trim($raw));
    $values = [];
    for ($i = 0; $i < $width; $i++) {
        $values[] = isset($parts[$i]) && is_numeric($parts[$i]) ? (float) $parts[$i] : 0.0;
    }
    return $values;
}

function startRun(RunManager $manager, array $body): array
{
    $problem = (string) ($body['problem'] ?? '');
    if ($problem === '') {
        return ['ok' => false, 'error' => 'problem required'];
    }
    return $manager->start($problem, $body['overrides'] ?? []);
}

function doStop(RunManager $manager): array
{
    $manager->stop();
    return ['ok' => true];
}

/**
 * Runtime health for the dashboard's banner: what a run launched now would get
 * (the server itself runs un-accelerated, so it predicts), plus the active run's
 * own self-report from meta.json if one exists.
 */
function runtimeHealth(SnapshotStore $store, RunManager $manager): array
{
    $meta = readJson($store->metaPath(runParam($manager)));
    $actual = is_array($meta) && isset($meta['runtime']) && is_array($meta['runtime']) ? $meta['runtime'] : null;
    return ['projected' => FastRuntime::projectedRunHealth(), 'actual' => $actual];
}

function runParam(RunManager $manager): string
{
    $run = $_GET['run'] ?? null;
    if (is_string($run) && preg_match('/^[A-Za-z0-9_-]+$/', $run)) {
        return $run;
    }
    return $manager->status()['active'] ?? 'run';
}

function streamRecords(SnapshotStore $store, string $run, int $from): array
{
    $path = $store->streamPath($run);
    $lines = is_file($path) ? file($path, FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES) : [];
    $total = count($lines);
    // A negative offset means "start at the live tail": the dashboard sends it on a
    // fresh page load so it follows the run from now on instead of replaying (and
    // re-downloading) the whole history. Return only the last record so the current
    // KPIs and champion network still render.
    if ($from < 0) {
        $from = max(0, $total - 1);
    }
    $truncated = $from > $total;
    if ($truncated) {
        $from = 0;
    }
    $records = [];
    for ($i = $from; $i < $total; $i++) {
        $decoded = json_decode($lines[$i], true);
        if ($decoded !== null) {
            $records[] = $decoded;
        }
    }
    // runId changes when a fresh run truncates and rewrites the stream, so the
    // dashboard can reset and follow the new run instead of getting stuck.
    $meta = readJson($store->metaPath($run));
    $runId = is_array($meta) ? ($meta['runId'] ?? null) : null;
    return ['run' => $run, 'records' => $records, 'next' => $total, 'truncated' => $truncated, 'runId' => $runId];
}

function readJson(string $path): mixed
{
    return is_file($path) ? json_decode((string) file_get_contents($path), true) : null;
}

function requestBody(): array
{
    $raw = file_get_contents('php://input') ?: '';
    $data = json_decode($raw, true);
    return is_array($data) ? $data : $_POST;
}

function contentType(string $file): string
{
    return match (strtolower(pathinfo($file, PATHINFO_EXTENSION))) {
        'html' => 'text/html; charset=utf-8',
        'js' => 'text/javascript; charset=utf-8',
        'css' => 'text/css; charset=utf-8',
        'json' => 'application/json',
        'svg' => 'image/svg+xml',
        default => 'application/octet-stream',
    };
}
