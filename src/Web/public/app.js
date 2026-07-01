'use strict';

const MAX_W = 8.388607;
const MAX_POINTS = 2000; // cap chart history so long runs stay light on refresh
const COL = {
    bio: '#3ad6a0', cyan: '#45c8ff', amber: '#ffcf6b', rose: '#ff6b81',
    violet: '#b07bff', line: '#1d2a36', muted: '#6b7d8c', ink: '#d7e3ea',
};

const state = {
    history: [],
    activeRun: null,
    offset: 0,
    freshLoad: true,    // a page (re)load follows the run from its live tail, not from the start
    running: false,
    runStarted: null,   // current run instance id (meta.runId); changes on a fresh Start
    staleRunId: null,   // after a fresh Start, ignore the old stream until the new instance appears
    staleSince: 0,      // when staleRunId was set, so a failed launch can't wedge the view forever
    prevEdges: null,    // connectionKey -> weight, for the change-flash diff
    problems: {},       // name -> {name, custom, inputs, outputs, memory, defaults}
    lastNet: null,      // last champion network record, for inference
    activations: null,  // node "type:index" -> value, set after a manual infer run
    activationName: 'sigmoid', // activation function of the inferred network (for the math tooltip)
    inferCount: -1,     // how many infer input boxes are rendered
    inferSeq: [],       // accumulated input steps for a memory network
    editing: false,     // editing an existing problem's data (vs creating new)
    cpMode: null,       // which button opened the create/edit panel: 'new' | 'data' | null
    cpRows: [],         // create/edit problem rows
    resultsShown: false, // results table already rendered for this (stopped) run
    agents: {},         // slug -> saved-agent summary (for the library dropdown)
    agentSeq: [],       // accumulated input steps for a loaded memory agent
};

// "parallel" is a checkbox: off = serial (the default, fastest for most problems);
// on evolves each island in its own worker process (coarse-grained island-parallel),
// which only pays off for genuinely heavy evaluation. Parallel width = island count.

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

// colour a match cell ("92%" or "ok"/"-") green / amber / rose by how good it is.
function matchClass(value) {
    const s = String(value);
    if (s === 'ok') return 'm-good';
    if (s === '-') return 'm-low';
    const pct = parseFloat(s);
    if (Number.isNaN(pct)) return '';
    return pct >= 80 ? 'm-good' : pct >= 50 ? 'm-mid' : 'm-low';
}

/* ---------- control panel ---------- */
// One source of truth for the config controls: input id -> the key used both in a
// problem's `defaults` and in the /api/start overrides. Adding a knob means adding
// one line here (plus the input in index.html and the override in RunManager).
const NUM_FIELDS = {
    cfgPopulation: 'population', cfgGenerations: 'generations', cfgIslands: 'islands', cfgSeed: 'seed',
    cfgCrossover: 'crossover', cfgWeightMut: 'weight-mutation', cfgAddNeuron: 'add-neuron',
    cfgAddConn: 'add-connection', cfgDelNeuron: 'remove-neuron', cfgDelConn: 'remove-connection',
    cfgSurvive: 'survive-rate', cfgElitism: 'elitism', cfgInitHidden: 'initial-hidden',
    cfgDiversity: 'diversity', cfgMigrateEvery: 'migration-every', cfgMigrateTop: 'migration-top',
    cfgSimplicity: 'simplicity',
    // weight-mutation mechanics
    cfgWeightCount: 'weight-count', cfgWeightAdjust: 'weight-adjust', cfgWeightRandomize: 'weight-randomize',
    // biology parameters (the toggles themselves live in BIO_FIELDS)
    cfgTraumaIntensity: 'trauma-intensity', cfgTraumaDecay: 'trauma-decay',
    cfgAdaptivePatience: 'adaptive-patience', cfgAdaptiveUp: 'adaptive-up', cfgAdaptiveDown: 'adaptive-down',
    cfgAdaptiveMin: 'adaptive-min', cfgAdaptiveMax: 'adaptive-max',
    cfgLifetimeSteps: 'lifetime-steps', cfgLifetimeStepSize: 'lifetime-step-size', cfgLamarckian: 'lamarckian',
};
const BIO_FIELDS = {
    bioTrauma: 'trauma', bioAdaptive: 'adaptive-mutation', bioLearning: 'lifetime-learning',
};

async function loadProblems() {
    let list = [];
    try { list = await (await fetch('/api/problems')).json(); } catch (_) { return; }
    const sel = $('problemSelect');
    const previous = sel.value;
    sel.innerHTML = '';
    state.problems = {};
    for (const p of list) {
        state.problems[p.name] = p;
        const opt = document.createElement('option');
        opt.value = p.name;
        opt.textContent = p.custom ? `★ ${p.name}` : p.name;
        sel.appendChild(opt);
    }
    const pick = state.problems[previous] ? previous : (list[0] && list[0].name);
    if (pick) { sel.value = pick; fillDefaults(pick); }
}

function fillDefaults(name) {
    const p = state.problems[name];
    if (!p) return;
    const d = p.defaults;
    for (const [id, key] of Object.entries(NUM_FIELDS)) $(id).value = d[key];
    for (const [id, key] of Object.entries(BIO_FIELDS)) $(id).checked = d[key];
    syncBiology();
    $('cfgActivation').value = d.activation;
    $('cfgHiddenLayers').value = d['hidden-layers'] || '';
    syncTopology();
    // A problem can prefer parallel (defaults.parallel); otherwise serial.
    $('cfgParallel').checked = !!d.parallel;
    // Sequence/memory group (moved into advanced params) reflects the selected problem.
    $('cpMemory').checked = !!p.memory;
    $('cpRandomize').checked = (p.window || 0) > 0;
    $('cpWindow').value = (p.window && p.window > 0) ? p.window : 5;
    $('cpPrime').value = p['window-prime'] || 0;
    syncCpRandom();
    $('deleteBtn').hidden = !p.custom;
    $('pmeta').innerHTML = `${p.inputs} in → ${p.outputs} out${p.memory ? ' · <span class="mem">memory</span>' : ''}`;
    $('problemDesc').textContent = p.description || '';
    syncParallel();
}

// Parallel needs 2+ islands and is incompatible with lifetime learning.
function syncParallel() {
    const off = $('bioLearning').checked || (+$('cfgIslands').value || 1) < 2;
    $('cfgParallel').disabled = off;
    if (off) $('cfgParallel').checked = false;
}

// A fixed layered topology ("hidden layers") seeds a frozen neuron count, so
// "init hidden" (a dynamic-topology knob) doesn't apply - grey it out when set.
function syncTopology() {
    const layered = $('cfgHiddenLayers').value.trim() !== '';
    $('cfgInitHidden').disabled = layered;
}

// Show each biology feature's parameters only when its toggle is on, and hide the
// group when off - so the "biology parameters" section only lists what's active.
function syncBiology() {
    const trauma = $('bioTrauma').checked;
    const adaptive = $('bioAdaptive').checked;
    const learning = $('bioLearning').checked;
    $('traumaParams').style.display = trauma ? '' : 'none';
    $('adaptiveParams').style.display = adaptive ? '' : 'none';
    $('lifetimeParams').style.display = learning ? '' : 'none';
    // A hint when nothing is on, so the empty section doesn't look broken.
    $('bioParamsEmpty').style.display = (trauma || adaptive || learning) ? 'none' : '';
}

// Random-window controls (the "sequence" group in advanced params) apply only to
// memory problems: the "randomize start" toggle shows when memory is on, and the
// window/prime sizes show once the toggle is on.
function syncCpRandom() {
    const memory = $('cpMemory').checked;
    const randomize = $('cpRandomize').checked;
    const show = memory && randomize;
    $('cpRandomizeField').style.display = memory ? '' : 'none';
    $('cpWindowField').style.display = show ? '' : 'none';
    $('cpPrimeField').style.display = show ? '' : 'none';
}

// Turning lifetime learning on while its step count is still 0 would be a no-op,
// so seed the established default - the user can still tune it down afterwards.
function syncLifetime() {
    if ($('bioLearning').checked && (+$('cfgLifetimeSteps').value || 0) <= 0) {
        $('cfgLifetimeSteps').value = 5;
    }
}

async function startRun(resume = false) {
    state.freshLoad = false; // an explicit Start/Continue follows the run from its first record
    const overrides = {
        // Parallel on => one worker per island, so send the island count (>=2 engages
        // island-parallel in RunManager; 1 island or off stays serial).
        parallel: $('cfgParallel').checked ? (+$('cfgIslands').value || 1) : 1,
        resume, activation: $('cfgActivation').value,
        'hidden-layers': $('cfgHiddenLayers').value.trim(),
    };
    for (const [id, key] of Object.entries(NUM_FIELDS)) overrides[key] = +$(id).value;
    for (const [id, key] of Object.entries(BIO_FIELDS)) overrides[key] = $(id).checked;
    // Sequence/memory controls (advanced params): memory toggles recurrent state; the
    // random scoring window is live only when "randomize start" is on, else window = 0.
    const seqWindow = $('cpMemory').checked && $('cpRandomize').checked;
    overrides.memory = $('cpMemory').checked;
    overrides.window = seqWindow ? Math.max(1, +$('cpWindow').value || 0) : 0;
    overrides['window-prime'] = seqWindow ? Math.max(0, +$('cpPrime').value || 0) : 0;
    // Continue: if the previous run still had generations left (it was stopped
    // early), finish that original plan and keep the same cap. Only once it has
    // reached its last specified generation does Continue run unbounded (∞).
    if (resume) {
        const last = state.history[state.history.length - 1];
        overrides.generations = (last && last.totalGenerations > 0 && last.gen < last.totalGenerations)
            ? last.totalGenerations - last.gen
            : 0;
    }
    const body = { problem: $('problemSelect').value, overrides };
    $('startBtn').disabled = true;
    try {
        const res = await (await fetch('/api/start', { method: 'POST', body: JSON.stringify(body) })).json();
        if (res.ok) {
            const previous = state.runStarted;
            switchRun(res.name);
            // A fresh Start truncates and rewrites the stream; ignore the old run's
            // leftover records until the new instance (new runId) shows up. A resume
            // appends to the same stream, so keep following it.
            state.staleRunId = resume ? null : previous;
            state.staleSince = Date.now();
            setStatus('live', 'starting...');
        }
    } catch (_) { /* ignore */ }
    setTimeout(() => { $('startBtn').disabled = false; }, 800);
}

// Continue keeps the current settings but resumes from the saved population.
const continueRun = () => startRun(true);

async function stopRun() {
    try { await fetch('/api/stop', { method: 'POST' }); } catch (_) { /* ignore */ }
    // Don't wait for the status poll to notice: give the run a beat to flush its last
    // checkpoint, then save its champion and show it in the predictions panel.
    setTimeout(finalizeStoppedRun, 800);
}

// Once a run is idle (finished or stopped), auto-save its champion and select it in
// the predictions panel so its results appear with no manual step. Idempotent.
async function finalizeStoppedRun() {
    if (state.resultsShown || !state.activeRun || !state.history.length) return;
    state.resultsShown = true;
    await autoSaveAgent();
}

// Clear everything tied to the records of one run instance (kept separate from
// switchRun so the poller can also reset in place when a run restarts).
function resetRunView() {
    state.history = [];
    state.offset = 0;
    state.prevEdges = null;
    state.lastNet = null;
    state.activations = null;
    state.inferCount = -1;
    state.inferSeq = [];
    state.resultsShown = false;
    $('infer').hidden = true;
}

function switchRun(name) {
    state.activeRun = name;
    state.runStarted = null;
    state.staleRunId = null;
    resetRunView();
}

/* ---------- following the active run ---------- */
async function statusLoop() {
    try {
        const s = await (await fetch('/api/status')).json();
        state.running = !!s.running;
        // Follow whatever run is active, so a page refresh lands on the running run.
        if (s.active && s.active !== state.activeRun) switchRun(s.active);
        $('startBtn').disabled = state.running;
        $('continueBtn').disabled = state.running;
        $('stopBtn').disabled = !state.running;
        updateHeader();
    } catch (_) { /* server not up yet */ }
    setTimeout(statusLoop, 1000);
}

// Surface a banner when the run environment isn't at full speed (OPcache/JIT off, or
// Xdebug slowing runs). Prefer the active run's own self-report (meta.runtime); before
// any run exists, fall back to what a run launched now would get (server projection).
async function healthLoop() {
    try {
        const h = await (await fetch('/api/health')).json();
        renderHealth(h);
    } catch (_) { /* ignore */ }
    setTimeout(healthLoop, 5000);
}

function renderHealth(h) {
    const banner = $('healthBanner');
    const info = h && (h.actual || h.projected);
    const warnings = (info && info.warnings) || [];
    if (!info || info.fast || warnings.length === 0) { banner.hidden = true; return; }
    const scope = h.actual ? 'This run is' : 'Runs will be';
    banner.innerHTML = `<b>⚠ ${scope} not running at full speed.</b> ` + warnings.map(esc).join(' ');
    banner.hidden = false;
}

async function poll() {
    let delay = 1200;
    try {
        if (state.activeRun) {
            // On a page (re)load, ask for the live tail (from=-1) so a long run never
            // replays its whole history into the chart - we adopt the current instance
            // and follow it from now on, keeping memory light.
            const from = state.freshLoad ? -1 : state.offset;
            const data = await (await fetch(`/api/stream?run=${encodeURIComponent(state.activeRun)}&from=${from}`)).json();

            if (state.freshLoad) {
                // Wait for the run instance id before committing, so we don't later
                // mistake it for a new run and reset back to a full replay.
                if (data.runId == null) { setTimeout(poll, 400); return; }
                state.freshLoad = false;
                state.runStarted = data.runId;
                state.offset = data.next;
                if (data.records.length) {
                    state.history.push(chartPoint(data.records[data.records.length - 1]));
                    renderRecord(data.records[data.records.length - 1]);
                    drawChart();
                }
                setTimeout(poll, state.running ? 350 : 1200);
                return;
            }

            // Just after a fresh Start, the old run's records may linger for a beat
            // before the engine truncates the stream. Wait for the new instance
            // (give up after a few seconds so a failed launch can't wedge the view).
            if (state.staleRunId != null && data.runId === state.staleRunId) {
                if (Date.now() - (state.staleSince || 0) < 8000) { setTimeout(poll, 250); return; }
                state.staleRunId = null;
            }
            state.staleRunId = null;

            // A new run instance (fresh Start) carries a new runId; follow it from
            // the top. A resume keeps the runId and just appends - no reset.
            if (data.runId != null && data.runId !== state.runStarted) {
                state.runStarted = data.runId;
                resetRunView();
                setTimeout(poll, 150);
                return;
            }
            if (data.truncated) { resetRunView(); setTimeout(poll, 150); return; }

            state.offset = data.next;
            if (data.records.length) {
                // Keep only lightweight chart points (not whole records with genomes),
                // capped, so a page refresh on a long run never replays a heavy history
                for (const rec of data.records) state.history.push(chartPoint(rec));
                if (state.history.length > MAX_POINTS) state.history = state.history.slice(-MAX_POINTS);
                // Render once per batch (the latest record) and draw the chart once. The
                // latest record always carries the current all-time-best network, so if any
                // record in the batch improved, redraw it - otherwise a fast run whose
                // improvement lands mid-batch would leave a stale champion on screen until a
                // Continue reset force-redrew it (looking like the network changed on resume).
                const batchImproved = data.records.some((r) => r.improved);
                renderRecord(data.records[data.records.length - 1], batchImproved);
                drawChart();
            }

            // Once the run is no longer live (finished or stopped), auto-save its best
            // champion to the agent library and select it - which renders its predictions
            // table and "try it" panel, exactly like finishing a run used to.
            if (!state.running && state.history.length && !state.resultsShown) {
                await finalizeStoppedRun();
            }
            delay = (state.running || data.records.length) ? 350 : 1200;
        }
    } catch (_) { delay = 1200; }
    setTimeout(poll, delay);
}

// lightweight point kept in history for the chart (no genome/islands payload)
const chartPoint = (rec) => ({ gen: rec.gen, best: rec.best, avg: rec.avg, allTimeBest: rec.allTimeBest, totalGenerations: rec.totalGenerations });

function renderRecord(rec, batchImproved = false) {
    const done = rec.totalGenerations > 0 && rec.gen >= rec.totalGenerations;
    setStatus(state.running ? 'live' : (done ? 'done' : 'live'), state.running ? `generation ${rec.gen}` : (done ? 'finished' : `generation ${rec.gen}`));

    // a resumed (Continue) run has no generation cap, so show the count over ∞
    setText('kpiGen', rec.totalGenerations > 0 ? `${rec.gen}/${rec.totalGenerations}` : `${rec.gen}/∞`);
    setText('kpiBest', fmt(rec.best));
    const matchPct = typeof rec.matchRate === 'number' ? `${Math.round(rec.matchRate * 100)}%` : '-';
    $('kpiAllTime').innerHTML = `${esc(fmt(rec.allTimeBest))} <span class="frac">/ ${matchPct}</span>`;
    setText('kpiAvg', fmt(rec.avg));
    setText('kpiPop', rec.popSize);
    setText('kpiMs', Math.round(rec.durationMs));
    setText('kpiElapsed', fmtDuration(rec.elapsedMs));

    // the champion network shows the all-time best, so only refresh it on improvement
    if (rec.improved || batchImproved || !state.lastNet) {
        setText('kpiHidden', rec.hidden);
        setText('kpiGenes', rec.genes);
        state.activations = null;
        drawNetwork(rec.network);
    }
    drawIslands(rec.islands || []);
}

function setStatus(cls, text) {
    $('dot').className = 'dot ' + cls;
    $('statusText').textContent = text;
}
// The header name (beside the app title) follows the live run while one is
// running, otherwise the problem currently selected in the control panel - so it
// never shows a stale run name after you pick a different problem.
function updateHeader() {
    const name = state.running && state.activeRun ? state.activeRun : ($('problemSelect').value || state.activeRun || '');
    if (name) {
        $('problem').textContent = name;
        document.title = `Rotifer - ${name}`;
    }
}

const setText = (id, v) => { $(id).textContent = v; };
const fmt = (v) => (typeof v === 'number' ? v.toFixed(4) : v);
// Whole-run duration (ms) as a compact human string: ms, s, m s, then h m.
function fmtDuration(ms) {
    if (typeof ms !== 'number' || !isFinite(ms)) return '-';
    if (ms < 1000) return `${Math.round(ms)}ms`;
    const s = ms / 1000;
    if (s < 60) return `${s.toFixed(1)}s`;
    const m = Math.floor(s / 60), rs = Math.round(s % 60);
    if (m < 60) return `${m}m ${rs}s`;
    const h = Math.floor(m / 60), rm = m % 60;
    return `${h}h ${rm}m`;
}

/* ---------- results + success rate ---------- */
// Render an "expected vs predicted" table (from a run champion or a loaded agent).
function renderResultsTable(table) {
    const hint = $('successHint');
    if (!table || !table.columns || !table.rows || !table.rows.length) {
        $('results').innerHTML = '';
        hint.textContent = '';
        hint.className = 'hint';
        return;
    }
    const matchCol = table.columns.findIndex((c) => c === 'match' || c === 'ok');
    const head = '<tr>' + table.columns.map((c) => `<th>${esc(c)}</th>`).join('') + '</tr>';
    const body = table.rows.map((r) => '<tr>' + r.map((c, i) => {
        const cls = i === matchCol ? ` class="${matchClass(c)}"` : '';
        return `<td${cls}>${esc(c)}</td>`;
    }).join('') + '</tr>').join('');
    $('results').innerHTML = `<table>${head}${body}</table>`;

    if (typeof table.successRate === 'number') {
        const pct = Math.round(table.successRate * 100);
        hint.textContent = `${pct}% success (how close predictions are to expected)`;
        hint.className = 'hint ' + (pct >= 80 ? 'good' : pct >= 50 ? 'mid' : 'low');
    } else {
        hint.textContent = '';
        hint.className = 'hint';
    }
}

/* ---------- fitness chart ---------- */
// Drop the accumulated points to free memory. The stream offset is untouched, so a
// live run simply keeps plotting from the next generation on.
function clearChart() {
    state.history = [];
    drawChart();
}

function drawChart() {
    const cv = $('chart');
    const ctx = cv.getContext('2d');
    const W = cv.width, H = cv.height, pad = 34;
    ctx.clearRect(0, 0, W, H);

    const h = state.history;
    if (h.length === 0) return;

    const series = { best: h.map((r) => r.best), avg: h.map((r) => r.avg), all: h.map((r) => r.allTimeBest) };
    let lo = Infinity, hi = -Infinity;
    for (const k in series) for (const v of series[k]) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    if (lo === hi) { hi += 1; lo -= 1; }

    const n = h.length;
    const x = (i) => pad + (n <= 1 ? 0 : (i / (n - 1)) * (W - pad * 2));
    const y = (v) => H - pad - ((v - lo) / (hi - lo)) * (H - pad * 2);

    ctx.strokeStyle = COL.line; ctx.lineWidth = 1; ctx.font = '11px JetBrains Mono, monospace'; ctx.fillStyle = COL.muted;
    ctx.textAlign = 'left';
    for (let g = 0; g <= 4; g++) {
        const gy = pad + (g / 4) * (H - pad * 2);
        ctx.beginPath(); ctx.moveTo(pad, gy); ctx.lineTo(W - pad, gy); ctx.stroke();
        ctx.fillText((hi - (g / 4) * (hi - lo)).toFixed(2), 2, gy + 3);
    }
    // x-axis: a few generation labels along the bottom so the time scale is legible.
    ctx.textAlign = 'center';
    const ticks = Math.min(6, n);
    for (let t = 0; t < ticks; t++) {
        const i = ticks <= 1 ? 0 : Math.round((t / (ticks - 1)) * (n - 1));
        ctx.fillText(String(h[i].gen), x(i), H - pad + 15);
    }
    ctx.fillText('generation', W / 2, H - 4);
    ctx.textAlign = 'left';
    line(ctx, series.all, x, y, COL.amber, 1.5);
    line(ctx, series.avg, x, y, COL.cyan, 1.5);
    line(ctx, series.best, x, y, COL.bio, 2.5, true);
}

function line(ctx, data, x, y, color, width, glow = false) {
    ctx.beginPath();
    data.forEach((v, i) => { const px = x(i), py = y(v); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); });
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineJoin = 'round';
    if (glow) { ctx.shadowColor = color; ctx.shadowBlur = 8; }
    ctx.stroke();
    ctx.shadowBlur = 0;
}

/* ---------- network graph ---------- */
function drawNetwork(net) {
    const svg = $('network');
    if (!net) { svg.innerHTML = ''; return; }
    state.lastNet = net;
    const W = 420, H = 320, pad = 28;

    // Lay the net out as tidy layers and route each wire around the neurons it would
    // otherwise cut across (see layoutNetwork): pos has every neuron's spot; routes maps
    // each connection to its path - endpoints plus the bend points that keep it off nodes;
    // hiddenLayers is how many hidden depth columns the topology formed.
    const { pos, routes, hiddenLayers } = layoutNetwork(net, W, H, pad);
    setText('kpiLayers', hiddenLayers);

    // Diff against the previous champion to flash what changed.
    const current = {};
    for (const g of net.genes) current[edgeKey(g)] = g[4];
    const prev = state.prevEdges;

    let edges = '', hits = '';
    for (const g of net.genes) {
        const ek = edgeKey(g);
        const pts = routes[ek];
        if (!pts || pts.length < 2) continue;
        let cls = '';
        if (prev) {
            if (!(ek in prev)) cls = 'flash-new';
            else if (Math.abs(prev[ek] - g[4]) > 0.05) cls = 'flash-change';
        }
        edges += path(pts, g[4], cls, ek);
        hits += hitPath(pts, ek, g[4], g); // transparent wide line so thin edges are easy to hover
    }

    let nodes = '';
    for (const [k, p] of Object.entries(pos)) {
        if (k[0] === 'd') continue; // routing bend point, not a neuron
        const type = +k.split(':')[0];
        const idx = k.split(':')[1];
        let fill = type === 1 ? COL.violet : COL.cyan;
        let valAttr = '';
        if (state.activations && k in state.activations) {
            fill = valueColor(state.activations[k]);
            valAttr = ` data-val="${state.activations[k].toFixed(3)}"`;
        }
        const r = type === 1 ? 6 : 7;
        const stroke = valAttr ? ' stroke="#fff" stroke-width="0.6"' : '';
        nodes += `<circle cx="${p.x}" cy="${p.y}" r="${r}" fill="${fill}"${stroke} opacity="0.95" data-node="${esc(label(type) + ' ' + idx)}" data-key="${esc(k)}"${valAttr}/>`;
    }

    svg.innerHTML = edges + hits + nodes;
    state.prevEdges = current;
    ensureInferInputs(net.inputs);
}

// A smooth left-to-right path through the endpoints and any routing bend points. A self-
// loop (a neuron wired to itself) becomes a rounded teardrop that leaves the neuron and
// curves straight back to it; a same-column hop between two neurons bows sideways so it
// doesn't lie on the node line.
function routePath(pts) {
    const a = pts[0], z = pts[pts.length - 1];
    if (a.x === z.x && a.y === z.y) {
        return `M${a.x},${a.y} C${a.x + 32},${a.y - 19} ${a.x + 32},${a.y + 19} ${a.x},${a.y}`;
    }
    let d = `M${a.x},${a.y}`;
    for (let i = 1; i < pts.length; i++) {
        const p = pts[i - 1], b = pts[i];
        const c1x = p.x === b.x ? p.x + 30 : (p.x + b.x) / 2;
        const c2x = p.x === b.x ? b.x + 30 : (p.x + b.x) / 2;
        d += ` C${c1x},${p.y} ${c2x},${b.y} ${b.x},${b.y}`;
    }
    return d;
}

function path(pts, w, cls, ek) {
    const col = w >= 0 ? COL.bio : COL.rose;
    const op = (0.2 + 0.7 * Math.min(1, Math.abs(w) / MAX_W)).toFixed(2);
    const sw = (0.5 + 2.4 * Math.min(1, Math.abs(w) / MAX_W)).toFixed(2);
    const cl = ('edge ' + cls).trim();
    return `<path class="${cl}" d="${routePath(pts)}" stroke="${col}" stroke-width="${sw}" fill="none" opacity="${op}" data-ek="${ek}"/>`;
}

function hitPath(pts, ek, w, g) {
    const lbl = `${label(g[0])} ${g[1]} → ${label(g[2])} ${g[3]}`;
    return `<path class="edge-hit" d="${routePath(pts)}" stroke="transparent" stroke-width="12" fill="none" data-ek="${ek}" data-w="${(+w).toFixed(3)}" data-l="${esc(lbl)}"/>`;
}

// Lay the network out as readable layers and route its wires so they don't run over
// neurons. This is the classic layered-graph recipe:
//  1. Layers - inputs on the left, outputs on the right, hidden neurons in depth columns
//     between them (one per fixed layer, or by longest path from the inputs when dynamic).
//  2. Routing points - every wire that skips a column gets an invisible bend point in each
//     column it crosses, so it can weave *between* those neurons instead of through them.
//  3. Untangle - each layer (neurons and bend points together) is repeatedly reordered to
//     its neighbours' average height and packed with a minimum gap, which lines connected
//     neurons up, pulls the long wires nearly straight, and minimises crossings.
// Inputs and outputs keep their index order as the anchors everything else settles against.
// Returns { pos: key -> {x,y} for every neuron and bend point, routes: edgeKey -> [points] }.
function layoutNetwork(net, W, H, pad) {
    const hidden = new Set();
    for (const g of net.genes) { if (g[0] === 1) hidden.add(g[1]); if (g[2] === 1) hidden.add(g[3]); }
    const hiddenArr = [...hidden].sort((a, b) => a - b);

    const layers = [];
    const layerOf = {};
    const addLayer = (keys) => { keys.forEach((k) => { layerOf[k] = layers.length; }); layers.push(keys.slice()); };
    addLayer([...Array(net.inputs).keys()].map((i) => node(0, i)));
    hiddenColumns(net, hiddenArr).forEach((col) => addLayer(col.map((h) => node(1, h))));
    addLayer([...Array(net.outputs).keys()].map((o) => node(2, o)));

    // Turn each wire into a chain of single-column hops, dropping a routing point in every
    // column a long (forward) wire skips. Same-column and recurrent wires stay direct.
    const routeKeys = {};
    let dummies = 0;
    for (const g of net.genes) {
        const u = node(g[0], g[1]), v = node(g[2], g[3]);
        const chain = [u];
        const lu = layerOf[u], lv = layerOf[v];
        if (lu !== undefined && lv !== undefined && lv - lu > 1) {
            for (let L = lu + 1; L < lv; L++) { const d = 'd' + dummies++; layerOf[d] = L; layers[L].push(d); chain.push(d); }
        }
        chain.push(v);
        routeKeys[edgeKey(g)] = chain;
    }

    // neighbours along the route chains (neurons + bend points), for the untangling
    const nbr = {};
    for (const ek in routeKeys) {
        const ch = routeKeys[ek];
        for (let i = 0; i < ch.length - 1; i++) {
            if (ch[i] === ch[i + 1]) continue; // a self-loop says nothing about layout height
            (nbr[ch[i]] ||= []).push(ch[i + 1]);
            (nbr[ch[i + 1]] ||= []).push(ch[i]);
        }
    }

    const L = layers.length;
    const top = pad, bottom = H - pad, gap = 26;
    const y = {};
    const initY = () => layers.forEach((lay) => lay.forEach((k, i) => { y[k] = lay.length <= 1 ? H / 2 : top + (i / (lay.length - 1)) * (bottom - top); }));
    initY();

    // settle one layer to its neighbours' mean height (optionally re-sorting by it): pack it
    // with a minimum gap so nothing overlaps, then centre the whole stack on the column's
    // mean desired height and clamp it in-frame. Centring is what keeps a small layer (e.g.
    // a fixed-layer network's column) in the middle instead of stuck against the top edge.
    const relax = (lay, sort) => {
        const want = {};
        for (const k of lay) { const ns = nbr[k]; want[k] = ns && ns.length ? ns.reduce((t, n) => t + y[n], 0) / ns.length : y[k]; }
        if (sort) lay.sort((a, b) => want[a] - want[b]);
        const n = lay.length;
        const g = n > 1 ? Math.min(gap, (bottom - top) / (n - 1)) : gap;
        let prev = -Infinity, meanWant = 0, meanY = 0;
        for (const k of lay) { y[k] = Math.max(want[k], prev + g); prev = y[k]; meanWant += want[k]; meanY += y[k]; }
        meanWant /= n; meanY /= n;
        const lo = y[lay[0]], hi = y[lay[n - 1]];
        if (hi - lo <= bottom - top) {
            // shift the stack so its mean sits at the mean desired height, kept in-frame
            const shift = Math.max(top - lo, Math.min(bottom - hi, meanWant - meanY));
            for (const k of lay) y[k] += shift;
        } else {
            // wants too spread to follow without leaving the frame: space evenly to fill it
            lay.forEach((k, i) => { y[k] = top + (i / (n - 1)) * (bottom - top); });
        }
    };
    // Inputs (first layer) and outputs (last layer) stay pinned to index order - neuron 0
    // on top, then 1, 2, ... down - so the graph always reads predictably. Only the hidden
    // columns get reordered to minimise crossings, so they never re-sort.
    const pinned = (li) => li === 0 || li === L - 1;
    const sweep = (sort) => { for (let li = 0; li < L; li++) relax(layers[li], sort && !pinned(li)); for (let li = L - 1; li >= 0; li--) relax(layers[li], sort && !pinned(li)); };

    const posIn = {};
    const reindex = () => layers.forEach((lay) => lay.forEach((k, i) => { posIn[k] = i; }));
    const countCrossings = () => {
        let c = 0;
        for (let li = 0; li < L - 1; li++) {
            const e = [];
            for (const k of layers[li]) for (const n of nbr[k] || []) if (layerOf[n] === li + 1) e.push([posIn[k], posIn[n]]);
            for (let i = 0; i < e.length; i++) for (let j = i + 1; j < e.length; j++) if ((e[i][0] - e[j][0]) * (e[i][1] - e[j][1]) < 0) c++;
        }
        return c;
    };
    const sideCross = (a, b, side) => { // crossings of a (above) vs b (below) into layer `side`
        let c = 0;
        for (const na of nbr[a] || []) if (layerOf[na] === side) for (const nb of nbr[b] || []) if (layerOf[nb] === side && posIn[nb] < posIn[na]) c++;
        return c;
    };
    const transpose = () => {
        for (let round = 0; round < 4; round++) {
            let improved = false;
            for (let li = 0; li < L; li++) {
                if (pinned(li)) continue; // keep inputs/outputs in index order
                const lay = layers[li];
                for (let i = 0; i < lay.length - 1; i++) {
                    const a = lay[i], b = lay[i + 1];
                    if (sideCross(b, a, li - 1) + sideCross(b, a, li + 1) < sideCross(a, b, li - 1) + sideCross(a, b, li + 1)) {
                        lay[i] = b; lay[i + 1] = a; posIn[a] = i + 1; posIn[b] = i; improved = true;
                    }
                }
            }
            if (!improved) break;
        }
    };

    // Minimise crossings by alternating a barycentre sort (order each layer by its
    // neighbours' average height) with a transpose pass (swap adjacent nodes when that
    // removes crossings), keeping the fewest-crossings ordering found. Only the hidden
    // columns reorder; inputs and outputs stay pinned to index order (see `pinned`), so the
    // graph reads top-to-bottom by neuron index even at the cost of a few extra crossings.
    let best = layers.map((l) => l.slice()), bestC = Infinity;
    for (let iter = 0; iter < 8 && bestC > 0; iter++) {
        sweep(true);
        reindex();
        transpose();
        const c = countCrossings();
        if (c < bestC) { bestC = c; best = layers.map((l) => l.slice()); }
    }
    for (let i = 0; i < L; i++) layers[i] = best[i];

    // final heights from the chosen order (no more re-sorting)
    initY();
    for (let iter = 0; iter < 6; iter++) sweep(false);

    // Stretch each layer to fill the full height, keeping its order and relative spacing -
    // so a small layer spreads out like a normal network instead of clustering in the middle
    // with empty space above and below. (Expanding the gaps can't create overlaps.)
    layers.forEach((lay) => {
        if (lay.length < 2) return;
        const ys = lay.map((k) => y[k]);
        const lo = Math.min(...ys), hi = Math.max(...ys), span = hi - lo;
        lay.forEach((k, i) => { y[k] = span > 1 ? top + ((y[k] - lo) / span) * (bottom - top) : top + (i / (lay.length - 1)) * (bottom - top); });
    });

    const pos = {};
    layers.forEach((lay, li) => {
        const x = li === 0 ? pad : li === L - 1 ? W - pad : pad + (li / (L - 1)) * (W - pad * 2);
        for (const k of lay) pos[k] = { x, y: y[k] };
    });

    const routes = {};
    for (const ek in routeKeys) routes[ek] = routeKeys[ek].map((k) => pos[k]);
    return { pos, routes, hiddenLayers: L - 2 }; // layers minus the input and output columns
}

// Group the present hidden neurons into ordered columns. A fixed layered network
// (net.layers = [5,3,5]) gets one column per layer, by the contiguous index ranges
// the engine assigns. Otherwise depth is computed from the wiring (longest path
// from the inputs), so dynamic networks still render as clean left-to-right layers.
function hiddenColumns(net, hiddenArr) {
    const layers = net.layers || [];
    if (layers.length) {
        const cols = layers.map(() => []);
        const layerOf = (idx) => {
            let sum = 0;
            for (let k = 0; k < layers.length; k++) { sum += layers[k]; if (idx < sum) return k; }
            return layers.length - 1;
        };
        for (const hIdx of hiddenArr) cols[layerOf(hIdx)].push(hIdx);
        return cols;
    }

    const incoming = {};
    for (const g of net.genes) (incoming[node(g[2], g[3])] ||= []).push(node(g[0], g[1]));
    const depth = {};
    for (let i = 0; i < net.inputs; i++) depth[node(0, i)] = 0;
    for (const hIdx of hiddenArr) { // ascending index = feed-forward order, so sources are ready
        let d = 1;
        for (const src of (incoming[node(1, hIdx)] || [])) if (depth[src] !== undefined) d = Math.max(d, depth[src] + 1);
        depth[node(1, hIdx)] = d;
    }
    const maxDepth = hiddenArr.reduce((m, hIdx) => Math.max(m, depth[node(1, hIdx)]), 0);
    const cols = [];
    for (let d = 1; d <= maxDepth; d++) cols.push(hiddenArr.filter((hIdx) => depth[node(1, hIdx)] === d));
    return cols;
}

const node = (t, i) => `${t}:${i}`;
const edgeKey = (g) => `${g[0]}:${g[1]}>${g[2]}:${g[3]}`;
const label = (t) => (t === 0 ? 'input' : t === 1 ? 'hidden' : 'output');

// neuron value -> colour: dark when quiet, green firing positive, rose negative.
function valueColor(v) {
    const t = Math.tanh(v);
    const target = t >= 0 ? [58, 214, 160] : [255, 107, 129];
    return mix([22, 32, 42], target, Math.min(1, Math.abs(t)));
}
const mix = (a, b, t) => `rgb(${a.map((x, i) => Math.round(x + (b[i] - x) * t)).join(',')})`;

/* ---------- hover: highlight a connection / read a neuron, with a styled tooltip ---------- */
let hotEdge = null;
function onEdgeOver(e) {
    const tip = $('edgeTip');
    const hit = e.target.closest('path.edge-hit');
    if (hit) {
        const vis = $('network').querySelector(`path.edge[data-ek="${hit.getAttribute('data-ek')}"]`);
        if (vis) { vis.classList.add('hot'); hotEdge = vis; }
        tip.innerHTML = `${esc(hit.getAttribute('data-l'))} &nbsp; weight <b>${esc(hit.getAttribute('data-w'))}</b>`;
        tip.hidden = false;
        positionTip(e);
        return;
    }
    const circle = e.target.closest('circle[data-node]');
    if (circle) {
        const val = circle.getAttribute('data-val');
        tip.innerHTML = val !== null
            ? neuronTip(circle.getAttribute('data-key'), circle.getAttribute('data-node'), parseFloat(val))
            : `${esc(circle.getAttribute('data-node'))} <span class="tdim">(run an input to see its value)</span>`;
        tip.hidden = false;
        positionTip(e);
    }
}

// Explain a neuron's value: activation( Σ incoming source×weight ). This is exactly
// how the engine computes it - each neuron is the activation of its weighted inputs.
function neuronTip(key, lbl, value) {
    const head = `${esc(lbl)} = <b>${value.toFixed(3)}</b>`;
    const net = state.lastNet;
    const [type, idx] = key.split(':').map(Number);
    if (type === 0) return `${head} <span class="tdim">(input value)</span>`;
    if (!net) return head;

    const incoming = net.genes.filter((g) => g[2] === type && g[3] === idx);
    if (!incoming.length) return head;

    let sum = 0;
    const rows = incoming.map((g) => {
        const sk = `${g[0]}:${g[1]}`;
        const sv = (state.activations && sk in state.activations) ? state.activations[sk] : 0;
        const term = sv * g[4];
        sum += term;
        return `<div class="trow">${esc(label(g[0]))} ${g[1]} <span class="tdim">(${sv.toFixed(3)})</span> × ${g[4].toFixed(3)} = <b>${term.toFixed(3)}</b></div>`;
    });
    return `${head}<div class="math">= ${esc(state.activationName)}(<b>${sum.toFixed(3)}</b>)${rows.join('')}</div>`;
}
function onEdgeOut(e) {
    if (!e.target.closest('path.edge-hit') && !e.target.closest('circle[data-node]')) return;
    if (hotEdge) { hotEdge.classList.remove('hot'); hotEdge = null; }
    $('edgeTip').hidden = true;
}
function onEdgeMove(e) { if (!$('edgeTip').hidden) positionTip(e); }
// Place the tooltip near the cursor, but flip it left/up when it would spill off
// the right or bottom edge of the window so it always stays fully visible.
function positionTip(e) {
    const tip = $('edgeTip');
    const gap = 14;
    const { offsetWidth: w, offsetHeight: h } = tip;
    let x = e.clientX + gap;
    let y = e.clientY + gap;
    if (x + w > window.innerWidth - 8) x = e.clientX - w - gap;
    if (y + h > window.innerHeight - 8) y = e.clientY - h - gap;
    tip.style.left = Math.max(8, x) + 'px';
    tip.style.top = Math.max(8, y) + 'px';
}

/* ---------- inference ("try it") ---------- */
const hasMemory = () => !!(state.problems[state.activeRun] && state.problems[state.activeRun].memory);

function ensureInferInputs(count) {
    $('infer').hidden = false;
    $('inferReset').hidden = !hasMemory();
    $('inferStep').hidden = !hasMemory();
    if (count === state.inferCount) return;
    state.inferCount = count;
    state.inferSeq = [];
    const wrap = $('inferInputs');
    wrap.innerHTML = '';
    for (let i = 0; i < count; i++) {
        const inp = document.createElement('input');
        // 0-1 is the suggested range (hint via min/max); other values are still accepted.
        inp.type = 'number'; inp.step = '0.1'; inp.min = '0'; inp.max = '1';
        inp.value = i === 0 ? '1' : '0';
        inp.title = `input ${i}`;
        wrap.appendChild(inp);
    }
}

async function runInference() {
    if (!state.activeRun || !state.lastNet) return;
    const vals = [...$('inferInputs').querySelectorAll('input')].map((i) => (i.value.trim() === '' ? '0' : i.value.trim()));

    // A memory network is fed one step at a time, accumulating; a plain one is independent.
    if (hasMemory()) {
        state.inferSeq.push(vals);
    } else {
        state.inferSeq = [vals];
    }
    const sequence = state.inferSeq.map((step) => step.join(',')).join(';');

    let res;
    try { res = await (await fetch(`/api/infer?run=${encodeURIComponent(state.activeRun)}&input=${encodeURIComponent(sequence)}`)).json(); } catch (_) { return; }
    if (!res || !res.ok) { $('inferOut').textContent = (res && res.error) || 'could not run'; return; }
    state.activations = res.nodes;
    state.activationName = res.activation || 'sigmoid';
    drawNetwork(state.lastNet);
    const outs = res.outputs.map((o, i) => `<span class="chip">y${i} = <b>${Number(o).toFixed(4)}</b></span>`).join('');
    $('inferOut').innerHTML = `output: ${outs}`;
    if (hasMemory()) $('inferStep').textContent = `step ${state.inferSeq.length}`;
}

function resetInferMemory() {
    state.inferSeq = [];
    state.activations = null;
    if (state.lastNet) drawNetwork(state.lastNet);
    $('inferStep').textContent = 'step 0';
    $('inferOut').textContent = '';
}

/* ---------- create / edit / delete problems ---------- */
function newProblem() {
    // Clicking New again while its panel is open closes it (toggle).
    if (!$('createPanel').hidden && state.cpMode === 'new') { closeCreatePanel(); return; }
    state.editing = false;
    state.cpMode = 'new';
    state.cpRows = [];
    $('cpName').value = '';
    $('cpInputs').value = 2;
    $('cpOutputs').value = 1;
    $('cpMemory').checked = false;
    $('cpRandomize').checked = false;
    $('cpWindow').value = 5;
    $('cpPrime').value = 0;
    syncCpRandom();
    $('cpDescription').value = '';
    $('cpTitle').innerHTML = 'New problem <span class="hint">give example inputs and the outputs you expect</span>';
    $('cpHint').textContent = 'Each row is one example. Values work best between 0 and 1. Defaults adapt to your data.';
    $('cpRunTemp').hidden = false; // New can run its rows once without saving
    $('cpError').textContent = '';
    cpShowEditor(true); // a fresh problem always edits rows (resets any episodic state)
    $('advParams').open = true; // the memory/window controls live there now
    $('createPanel').hidden = false;
    cpAddRow(); cpAddRow();
}

function closeCreatePanel() {
    $('createPanel').hidden = true;
    state.editing = false;
    state.cpMode = null;
}

// View a problem's data and let the user edit it; saving makes a separate custom copy.
async function viewData() {
    // Clicking Data again while its panel is open closes it (toggle).
    if (!$('createPanel').hidden && state.cpMode === 'data') { closeCreatePanel(); return; }
    await loadDataPanel($('problemSelect').value);
}

// Fetch and render one problem's data into the panel (no toggle), so changing the
// selected problem can refresh an already-open Data view in place.
async function loadDataPanel(name) {
    if (!name) return;
    let d;
    try { d = await (await fetch(`/api/problemdata?name=${encodeURIComponent(name)}`)).json(); } catch (_) { return; }
    if (!d || !d.ok) return;
    state.editing = true;
    state.cpMode = 'data';
    state.cpRows = d.rows.map((r) => ({ input: r.input.slice(), output: r.output.slice() }));
    if (!d.episodic && state.cpRows.length === 0) state.cpRows = [{ input: Array(d.inputs).fill(0), output: Array(d.outputs).fill(0) }];
    $('cpName').value = d.custom ? d.name.replace(/^custom_/, '') : d.name;
    $('cpInputs').value = d.inputs;
    $('cpOutputs').value = d.outputs;
    // The memory / window controls live in advanced params and already reflect this
    // problem (fillDefaults set them on selection); the Data view must NOT reset them,
    // so a user's edits survive opening/closing it. Save reads them from there.
    $('cpDescription').value = d.description || '';
    $('cpTitle').innerHTML = `Edit '${esc(d.name)}' <span class="hint">saves a separate custom copy; the original is untouched</span>`;
    $('cpRunTemp').hidden = true; // the Data view edits/saves; it does not run
    $('cpError').textContent = '';
    // An episodic problem (e.g. flappy_bird) has no dataset - it simulates each
    // evaluation - so hide the row editor instead of showing a misleading lone row.
    cpShowEditor(!d.episodic);
    $('cpHint').textContent = d.episodic
        ? 'This problem has no dataset - it runs a live simulation each evaluation, so there are no input/output rows to edit.'
        : 'Edit the values, then Save a custom copy.';
    $('advParams').open = true; // the memory/window controls live there now
    $('createPanel').hidden = false;
    if (!d.episodic) cpRender();
}

// Show or hide the row editor (rows + add-row buttons + Save). Hidden for episodic
// problems that have no editable dataset. The rows container has an explicit
// `display:flex` that would override the [hidden] attribute, so toggle its display
// directly and clear stale rows when hiding.
function cpShowEditor(show) {
    $('cpAddRowTop').hidden = !show;
    $('cpAddRowBottom').hidden = !show;
    $('cpSave').hidden = !show;
    $('cpRows').style.display = show ? '' : 'none';
    if (!show) $('cpRows').innerHTML = '';
}

function cpDims() {
    return {
        inputs: Math.max(1, +$('cpInputs').value || 1),
        outputs: Math.max(1, +$('cpOutputs').value || 1),
    };
}

function cpAddRow() {
    const { inputs, outputs } = cpDims();
    state.cpRows.push({ input: Array(inputs).fill(0), output: Array(outputs).fill(0) });
    cpRender();
}

function cpRender() {
    const { inputs, outputs } = cpDims();
    const resize = (arr, n) => { const o = arr.slice(0, n); while (o.length < n) o.push(0); return o; };
    state.cpRows.forEach((r) => { r.input = resize(r.input, inputs); r.output = resize(r.output, outputs); });

    // 0-1 is suggested (min/max hint) but other values are still accepted.
    const cell = (kind, ri, ci, v) =>
        `<input type="number" step="0.1" min="0" max="1" data-kind="${kind}" data-row="${ri}" data-col="${ci}" value="${v}">`;

    $('cpRows').innerHTML = state.cpRows.map((r, ri) => {
        const ins = r.input.map((v, ci) => cell('in', ri, ci, v)).join('');
        const outs = r.output.map((v, ci) => cell('out', ri, ci, v)).join('');
        return `<div class="cp-row"><span class="cp-label">#${ri + 1}</span><div class="grp">${ins}</div>`
            + `<span class="arrow">→</span><div class="grp">${outs}</div>`
            + `<button class="cp-del" data-row="${ri}" title="remove row">×</button></div>`;
    }).join('');

    $('cpRows').querySelectorAll('input[data-kind]').forEach((inp) => {
        inp.addEventListener('input', () => {
            const ri = +inp.dataset.row, ci = +inp.dataset.col;
            state.cpRows[ri][inp.dataset.kind === 'in' ? 'input' : 'output'][ci] = inp.value;
        });
    });
    $('cpRows').querySelectorAll('.cp-del').forEach((btn) => {
        btn.addEventListener('click', () => { state.cpRows.splice(+btn.dataset.row, 1); cpRender(); });
    });
}

async function cpSave(runAfter) {
    const { inputs, outputs } = cpDims();
    // Random scoring window: only for a memory problem with "randomize start" on.
    const randomize = $('cpMemory').checked && $('cpRandomize').checked;
    const body = {
        name: $('cpName').value,
        inputs, outputs,
        memory: $('cpMemory').checked,
        window: randomize ? Math.max(1, +$('cpWindow').value || 0) : 0,
        'window-prime': randomize ? Math.max(0, +$('cpPrime').value || 0) : 0,
        description: $('cpDescription').value,
        rows: state.cpRows.map((r) => ({ input: r.input.map(Number), output: r.output.map(Number) })),
    };
    let res;
    try { res = await (await fetch('/api/problems/create', { method: 'POST', body: JSON.stringify(body) })).json(); } catch (_) { $('cpError').textContent = 'request failed'; return; }
    if (!res.ok) { $('cpError').textContent = res.error || 'invalid definition'; return; }
    $('cpError').textContent = '';
    $('createPanel').hidden = true;
    state.cpMode = null;
    state.cpRows = [];

    // Editing existing data must not reset the user's tuning; only a brand-new
    // problem loads its recommended defaults.
    const keep = state.editing ? snapshotControls() : null;
    await loadProblems();
    $('problemSelect').value = res.name;
    if (keep) { restoreControls(keep); markSelected(res.name); } else { fillDefaults(res.name); }
    state.editing = false;
    if (runAfter) startRun();
}

function snapshotControls() {
    const snap = {
        parallel: $('cfgParallel').checked, activation: $('cfgActivation').value, hiddenLayers: $('cfgHiddenLayers').value,
        // Sequence/memory controls are advanced params too, so editing a problem's data
        // and saving must leave them exactly as the user set them (not reset to stored).
        cpMemory: $('cpMemory').checked, cpRandomize: $('cpRandomize').checked,
        cpWindow: $('cpWindow').value, cpPrime: $('cpPrime').value,
    };
    for (const id in NUM_FIELDS) snap[id] = $(id).value;
    for (const id in BIO_FIELDS) snap[id] = $(id).checked;
    return snap;
}

function restoreControls(c) {
    $('cfgParallel').checked = c.parallel;
    $('cfgActivation').value = c.activation;
    $('cfgHiddenLayers').value = c.hiddenLayers;
    $('cpMemory').checked = c.cpMemory;
    $('cpRandomize').checked = c.cpRandomize;
    $('cpWindow').value = c.cpWindow;
    $('cpPrime').value = c.cpPrime;
    for (const id in NUM_FIELDS) $(id).value = c[id];
    for (const id in BIO_FIELDS) $(id).checked = c[id];
    syncParallel();
    syncCpRandom();
    syncTopology();
    syncBiology();
}

// Update only the per-problem badge + delete button, leaving the tuning fields alone.
function markSelected(name) {
    const p = state.problems[name];
    if (!p) return;
    $('deleteBtn').hidden = !p.custom;
    $('pmeta').innerHTML = `${p.inputs} in → ${p.outputs} out${p.memory ? ' · <span class="mem">memory</span>' : ''}`;
    $('problemDesc').textContent = p.description || '';
}

async function deleteProblem() {
    const name = $('problemSelect').value;
    if (!name || !(state.problems[name] && state.problems[name].custom)) return;
    try { await fetch('/api/problems/delete', { method: 'POST', body: JSON.stringify({ name }) }); } catch (_) { /* ignore */ }
    await loadProblems();
}

/* ---------- saved agents (auto-saved champions, integrated into the predictions panel) ---------- */
// Refresh the dropdown from the server, keeping the current selection where possible,
// then show whichever agent ends up selected.
async function loadAgents(keep) {
    let list = [];
    try { list = await (await fetch('/api/agents')).json(); } catch (_) { return; }
    if (!Array.isArray(list)) list = [];
    state.agents = {};
    const sel = $('agentSelect');
    const previous = keep || sel.value;
    sel.innerHTML = '';
    for (const a of list) {
        state.agents[a.slug] = a;
        const opt = document.createElement('option');
        opt.value = a.slug;
        // Show the on-disk path in the dropdown itself.
        opt.textContent = a.path || a.name;
        sel.appendChild(opt);
    }
    $('agentEmpty').hidden = list.length > 0;
    $('agentDeleteBtn').style.display = list.length ? '' : 'none';
    $('agentSelect').parentElement.style.display = list.length ? '' : 'none';
    if (list.length) {
        sel.value = state.agents[previous] ? previous : list[0].slug;
        await showAgent(sel.value);
    } else {
        $('agentRun').hidden = true;
        $('agentMeta').textContent = '';
        $('results').innerHTML = '';
        $('successHint').textContent = '';
    }
}

// Show one agent: its provenance (incl. genes count + saved path), its predictions
// table (rebuilt server-side over the problem's data), and the "try it" inputs.
async function showAgent(slug) {
    const a = state.agents[slug];
    if (!a) { $('agentRun').hidden = true; $('agentMeta').textContent = ''; return; }
    const match = typeof a.matchRate === 'number' ? ` · match <b>${Math.round(a.matchRate * 100)}%</b>` : '';
    const mem = a.memory ? ' · <span class="mem">memory</span>' : '';
    $('agentMeta').innerHTML = `<b>${esc(a.problem || '')}</b> · ${a.inputs} in → ${a.outputs} out${mem}`
        + ` · fitness <b>${Number(a.fitness).toFixed(4)}</b>${match}`
        + ` · genes <b>${a.geneCount}</b> · hidden <b>${a.hidden}</b>`;

    // Rebuild the expected-vs-predicted table for this agent (same view a finished run shows).
    let table = null;
    try { table = await (await fetch(`/api/agents/predictions?name=${encodeURIComponent(slug)}`)).json(); } catch (_) { /* ignore */ }
    renderResultsTable(table && table.ok ? table : null);

    buildAgentInputs(a);
    $('agentRun').hidden = false;
    $('agentReset').hidden = !a.memory;
    $('agentStep').hidden = !a.memory;
    $('agentOut').textContent = '';
}

function buildAgentInputs(a) {
    state.agentSeq = [];
    const wrap = $('agentInputs');
    wrap.innerHTML = '';
    for (let i = 0; i < a.inputs; i++) {
        const inp = document.createElement('input');
        inp.type = 'number'; inp.step = '0.1'; inp.min = '0'; inp.max = '1';
        inp.value = i === 0 ? '1' : '0';
        inp.title = `input ${i}`;
        wrap.appendChild(inp);
    }
    if (a.memory) $('agentStep').textContent = 'step 0';
}

async function runAgentInference() {
    const slug = $('agentSelect').value;
    const a = state.agents[slug];
    if (!a) return;
    const vals = [...$('agentInputs').querySelectorAll('input')].map((i) => (i.value.trim() === '' ? '0' : i.value.trim()));
    // A memory agent is fed one step at a time, accumulating; a plain one is independent.
    if (a.memory) state.agentSeq.push(vals); else state.agentSeq = [vals];
    const sequence = state.agentSeq.map((step) => step.join(',')).join(';');

    let res;
    try { res = await (await fetch(`/api/agents/infer?name=${encodeURIComponent(slug)}&input=${encodeURIComponent(sequence)}`)).json(); } catch (_) { return; }
    if (!res || !res.ok) { $('agentOut').textContent = (res && res.error) || 'could not run'; return; }
    const outs = res.outputs_values.map((o, i) => `<span class="chip">y${i} = <b>${Number(o).toFixed(4)}</b></span>`).join('');
    $('agentOut').innerHTML = `output: ${outs}`;
    if (a.memory) $('agentStep').textContent = `step ${state.agentSeq.length}`;
}

function resetAgentMemory() {
    state.agentSeq = [];
    $('agentStep').textContent = 'step 0';
    $('agentOut').textContent = '';
}

// Auto-save the champion when a run ends (finished or stopped), under the run name,
// then reload the library and select the freshly-saved agent.
async function autoSaveAgent() {
    if (!state.activeRun) return;
    try {
        const res = await (await fetch('/api/agents/save', { method: 'POST', body: JSON.stringify({ run: state.activeRun }) })).json();
        if (res && res.ok) await loadAgents(res.name);
    } catch (_) { /* ignore */ }
}

async function deleteAgentSelected() {
    const slug = $('agentSelect').value;
    if (!slug) return;
    try { await fetch('/api/agents/delete', { method: 'POST', body: JSON.stringify({ name: slug }) }); } catch (_) { /* ignore */ }
    await loadAgents();
}

/* ---------- islands ---------- */
function drawIslands(islands) {
    const root = $('islands');
    const best = Math.max(1e-9, ...islands.map((i) => i.best));
    root.innerHTML = islands.map((isl) => {
        const heat = Math.min(1, isl.best / best);
        const glow = `background: radial-gradient(120px 80px at 50% 0%, rgba(58,214,160,${(0.10 + 0.5 * heat).toFixed(2)}), transparent)`;
        const trauma = Math.min(100, Math.round((isl.trauma || 0) * 100));
        return `<div class="island">
            <div class="glow" style="${glow}"></div>
            <h3><span data-tip="A separate sub-population. It evolves on its own and trades its best with neighbours.">island ${isl.index}</span><span data-tip="Organisms living here.">${isl.size} pop</span></h3>
            <div class="big" data-tip="Best fitness in this island.">${isl.best.toFixed(4)}</div>
            <div class="row" data-tip="How strongly this island is mutating right now. Above 1 = exploring harder because progress stalled."><span>mutation</span><span>${(isl.mutationScale || 1).toFixed(2)}×</span></div>
            <div class="row" data-tip="Inherited 'stress' level. Hardship raises it; it makes offspring mutate more, then fades over generations. 0% = calm."><span>stress (trauma)</span><span>${trauma}%</span></div>
            <div class="bar"><i style="width:${trauma}%"></i></div>
        </div>`;
    }).join('');
}

/* ---------- wiring ---------- */
async function init() {
    await loadProblems();
    $('startBtn').addEventListener('click', () => startRun(false));
    $('continueBtn').addEventListener('click', continueRun);
    $('stopBtn').addEventListener('click', stopRun);
    $('problemSelect').addEventListener('change', () => {
        const name = $('problemSelect').value;
        fillDefaults(name);
        updateHeader();
        // Keep an open Data view in sync with the selected problem.
        if (!$('createPanel').hidden && state.cpMode === 'data') loadDataPanel(name);
    });
    $('bioTrauma').addEventListener('change', syncBiology);
    $('bioAdaptive').addEventListener('change', syncBiology);
    $('bioLearning').addEventListener('change', () => { syncParallel(); syncLifetime(); syncBiology(); });
    $('cfgIslands').addEventListener('input', syncParallel);
    $('cfgHiddenLayers').addEventListener('input', syncTopology);
    $('newBtn').addEventListener('click', newProblem);
    $('viewBtn').addEventListener('click', viewData);
    $('deleteBtn').addEventListener('click', deleteProblem);
    $('cpAddRowTop').addEventListener('click', () => cpAddRow());
    $('cpAddRowBottom').addEventListener('click', () => cpAddRow());
    $('cpSave').addEventListener('click', () => cpSave(false));
    $('cpRunTemp').addEventListener('click', () => cpSave(true));
    $('cpCancel').addEventListener('click', closeCreatePanel);
    $('cpInputs').addEventListener('change', cpRender);
    $('cpOutputs').addEventListener('change', cpRender);
    $('cpMemory').addEventListener('change', syncCpRandom);
    $('cpRandomize').addEventListener('change', syncCpRandom);
    $('inferBtn').addEventListener('click', runInference);
    $('inferReset').addEventListener('click', resetInferMemory);
    $('clearChartBtn').addEventListener('click', clearChart);
    $('agentDeleteBtn').addEventListener('click', deleteAgentSelected);
    $('agentSelect').addEventListener('change', () => showAgent($('agentSelect').value));
    $('agentRunBtn').addEventListener('click', runAgentInference);
    $('agentReset').addEventListener('click', resetAgentMemory);

    const svg = $('network');
    svg.addEventListener('mouseover', onEdgeOver);
    svg.addEventListener('mouseout', onEdgeOut);
    svg.addEventListener('mousemove', onEdgeMove);

    loadAgents();
    statusLoop();
    healthLoop();
    poll();
}

init();
