#!/usr/bin/env node
/**
 * G3 V8b driver: the new `hard` against today's `hard`, on the product stack, in one process.
 *
 *   node tests/htb_integration/g3_match.mjs bind [--write]
 *   node tests/htb_integration/g3_match.mjs run smoke --python <python>
 *   node tests/htb_integration/g3_match.mjs run match --python <python> --kill-s <seconds>
 *
 * Card: docs/superpowers/2026-11-09-htb-g3-match-card.md. Run only through a stage's wrapper
 * (docs/superpowers/evidence/htb-g3-{smoke,match}-run/01_wrapper.sh), which enforces the idle gate, the
 * background-compute check, the ceiling and the gate restoration.
 *
 * Players, each built from the product's own code and nothing else:
 *   - new hard: resolvePolicy('hard', HARD_MODE protocol) -> protocolMove (server/protocol_move.js) on the pinned
 *     student 78c16a31c3dba37f, its session wrapped in MirrorAveragingInference, exactly as server/index.js builds it.
 *     The one difference from a served request: `seeds`, protocolMove's test-only argument, is derived per move from
 *     MATCH_SEED (moveSeeds) instead of drawn fresh, so every game is reproducible and fresh streams per request are
 *     kept, as in the product.
 *   - today's hard: resolvePolicy('hard', HARD_MODE legacy) -> plain MCTS with the policy's simulations, then
 *     selectMoveForRequest, on the default model c34b7ff3297c785a: server/index.js's computeBestMove plain branch.
 *   Each move's state goes through toDict/fromDict, as a request does.
 *
 * Games: the empty 24x24 board, red first, to state.isTerminal() (the product's game-end rule; no 280 cap). Pair p,
 * game 0: new hard is red; game 1: new hard is black.
 *
 * It decides nothing. g3_analyse.mjs, which imports none of this file, validates the evidence, derives the timing
 * ceiling from the smoke and the verdict from the match.
 *
 * Writes, create-only, in docs/superpowers/evidence/htb-g3-<stage>/: games/pair_PPPP_game_G.json per game,
 * progress.jsonl (one line per game, after its sidecar and its background-compute check), record.json at the end.
 * Exit: 0 COMPLETED, 2 USAGE, 3 VOID (terminated, background compute, error), 4 REFUSED (before any game).
 */
import { spawnSync, execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync, appendFileSync } from 'node:fs';
import { dirname, join, normalize, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { performance } from 'node:perf_hooks';

import { TwixtState } from '../../server/gameLogic.js';
import { AlphaZeroInference } from '../../server/inference.js';
import { MCTS } from '../../server/mcts.js';
import { MirrorAveragingInference } from '../../server/mirror_inference.js';
import { assertSessionContract, resolveHardModel, resolveModel } from '../../server/model_manifest.js';
import { protocolMove } from '../../server/protocol_move.js';
import { resolvePolicy, selectMoveForRequest } from '../../server/readout_policy.js';
import * as GATES from './g3_authorization.mjs';

export const REPO_ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
export const SCHEMA = 'htb-g3-game/1';
export const RECORD_SCHEMA = 'htb-g3-record/1';
export const BINDING_SCHEMA = 'htb-g3-binding/1';
export const BINDING_RELPATH = 'tests/htb_integration/g3_binding.json';
export const DRIVER_RELPATH = 'tests/htb_integration/g3_match.mjs';
export const ANALYSER_RELPATH = 'tests/htb_integration/g3_analyse.mjs';
export const GATE_RELPATH = 'tests/htb_integration/g3_authorization.mjs';

/** Card §4: the seed namespace. Fixed before any game; never used by an earlier run. */
export const MATCH_SEED = 20261109;
/** Card §3: one colour-reversed pair for the smoke, 110 fresh pairs for the match. */
export const PAIRS = Object.freeze({ smoke: 1, match: 110 });
export const GATE_FOR = Object.freeze({ smoke: 'G3_SMOKE_AUTHORIZED', match: 'G3_MATCH_AUTHORIZED' });
export const NEW_MODEL_ID = '78c16a31c3dba37f';
export const LEGACY_MODEL_ID = 'c34b7ff3297c785a';
export const TIMING_RELPATH = 'docs/superpowers/evidence/htb-g3-smoke/timing.json';
export const outDirRel = (stage) => `docs/superpowers/evidence/htb-g3-${stage}`;
export const sidecarName = (pair, game) => `pair_${String(pair).padStart(4, '0')}_game_${game}.json`;

export const EXIT = Object.freeze({ COMPLETED: 0, USAGE: 2, VOID: 3, REFUSED: 4 });

/** Files the driver executes besides its import closure (card §7). */
export const EXTRA_FILES = Object.freeze([
  'package.json',
  'package-lock.json',
  'scripts/GPU/__init__.py',
  'scripts/GPU/alphazero/__init__.py',
  'scripts/GPU/alphazero/background_compute.py',
  'scripts/GPU/alphazero/idle_gate.py',
  'docs/superpowers/2026-11-09-htb-g3-match-card.md',
  'docs/superpowers/evidence/htb-g3-smoke-run/01_wrapper.sh',
  'docs/superpowers/evidence/htb-g3-match-run/01_wrapper.sh',
]);
export const MODEL_MANIFESTS = Object.freeze({
  new_hard: `models/${NEW_MODEL_ID}/manifest.json`,
  legacy_hard: `models/${LEGACY_MODEL_ID}/manifest.json`,
});
export const ORT_FILES = Object.freeze([
  'node_modules/onnxruntime-node/package.json',
  'node_modules/onnxruntime-common/package.json',
  'node_modules/onnxruntime-node/bin/napi-v6/darwin/arm64/onnxruntime_binding.node',
  'node_modules/onnxruntime-node/bin/napi-v6/darwin/arm64/libonnxruntime.1.23.2.dylib',
]);

export class G3Refusal extends Error {
  constructor(code, message) {
    super(message);
    this.name = 'G3Refusal';
    this.code = code;
  }
}
export class G3Void extends Error {
  constructor(step, message) {
    super(message);
    this.name = 'G3Void';
    this.step = step;
  }
}
const refuse = (code, msg) => {
  throw new G3Refusal(code, msg);
};

const sha256 = (buf) => createHash('sha256').update(buf).digest('hex');
const fileSha = (root, rel) => sha256(readFileSync(join(root, rel)));

// ---------------------------------------------------------------------------------------------------------------
// Seeds (card §4)
// ---------------------------------------------------------------------------------------------------------------

/**
 * The new hard's two streams for one move: sha256 of the move's identity, split into two 48-bit integers (the
 * range protocolMove's own fresh seeds use). Today's hard draws no randomness.
 */
export function moveSeeds(stage, pair, game, ply) {
  const h = sha256(`htb-g3/${MATCH_SEED}/${stage}/pair ${pair}/game ${game}/ply ${ply}`);
  return { search: parseInt(h.slice(0, 12), 16), readout: parseInt(h.slice(12, 24), 16) };
}

// ---------------------------------------------------------------------------------------------------------------
// Players
// ---------------------------------------------------------------------------------------------------------------

/** New hard: the product's protocol move, with the move's derived seeds. */
export function makeNewHardPlayer({ inference, modelId }) {
  const policy = resolvePolicy({ difficulty: 'hard', hardMode: 'protocol' });
  return {
    label: 'new_hard',
    modelId,
    policy,
    async move(state, { signal, seeds }) {
      const gameState = TwixtState.fromDict(state.toDict());
      const out = await protocolMove({ gameState, policy, inference, signal, seeds });
      return { moveKey: out.moveKey, nSimulations: out.nSimulations, search: 'protocol', valueKind: out.valueKind };
    },
  };
}

/** Today's hard: server/index.js computeBestMove's plain branch under HARD_MODE=legacy. */
export function makeLegacyHardPlayer({ inference, modelId }) {
  const policy = resolvePolicy({ difficulty: 'hard', hardMode: 'legacy' });
  return {
    label: 'legacy_hard',
    modelId,
    policy,
    async move(state, { signal }) {
      const gameState = TwixtState.fromDict(state.toDict());
      const mcts = new MCTS(inference, { nSimulations: policy.nSims });
      const { visitCounts } = await mcts.search(gameState, { signal, progressEvery: 50, progressMinMs: 100 });
      if (signal?.aborted) throw new G3Void('terminated', 'search aborted');
      if (!visitCounts || visitCounts.size === 0) throw new Error('Search aborted');
      const { moveKey } = selectMoveForRequest({
        visitCounts,
        difficulty: 'hard',
        hardMode: 'legacy',
        selectMove: (counts, temp) => mcts.selectMove(counts, temp),
      });
      return { moveKey, nSimulations: policy.nSims, search: 'plain', valueKind: null };
    },
  };
}

// ---------------------------------------------------------------------------------------------------------------
// One game
// ---------------------------------------------------------------------------------------------------------------

export const terminationOf = (state) =>
  state.winner() !== null ? 'win' : state.legalMoves().length === 0 ? 'no_legal_moves' : 'max_plies';

/**
 * Play one game from the empty board to state.isTerminal(). `now` is a monotonic millisecond clock.
 * Returns the moves, per-move records and the result; times every move call.
 */
export async function playGame({ newPlayer, legacyPlayer, newIsRed, stage, pair, game, signal, now = () => performance.now() }) {
  let state = new TwixtState({});
  const moves = [];
  const perMove = [];
  while (!state.isTerminal()) {
    if (signal?.aborted) throw new G3Void('terminated', `stopped at ply ${state.ply}`);
    const newToMove = (state.toMove === 'red') === newIsRed;
    const player = newToMove ? newPlayer : legacyPlayer;
    const seeds = newToMove ? moveSeeds(stage, pair, game, state.ply) : null;
    const t0 = now();
    let out;
    try {
      out = await player.move(state, { signal, seeds });
    } catch (err) {
      if (signal?.aborted) throw new G3Void('terminated', `stopped at ply ${state.ply}`);
      throw err;
    }
    const ms = now() - t0;
    if (signal?.aborted) throw new G3Void('terminated', `stopped at ply ${state.ply}`);
    const mv = out.moveKey.split(',').map(Number);
    if (!state.legalMoves().some((m) => m[0] === mv[0] && m[1] === mv[1])) {
      throw new Error(`${player.label} returned an illegal move ${out.moveKey} at ply ${state.ply}`);
    }
    perMove.push({
      ply: state.ply,
      player: player.label,
      ms,
      n_simulations: out.nSimulations,
      search: out.search,
      value_kind: out.valueKind,
      seeds,
    });
    moves.push(mv);
    state = state.applyMove(mv);
  }
  const result = state.gameResult();
  const newColour = newIsRed ? 'red' : 'black';
  return {
    moves,
    perMove,
    result,
    termination: terminationOf(state),
    plyCount: state.ply,
    newScore: result === 'draw' ? 0.5 : result === newColour ? 1 : 0,
    moveMsTotal: perMove.reduce((a, m) => a + m.ms, 0),
  };
}

// ---------------------------------------------------------------------------------------------------------------
// One stage
// ---------------------------------------------------------------------------------------------------------------

const writeJsonAtomic = (path, obj) => {
  const tmp = `${path}.tmp`;
  writeFileSync(tmp, JSON.stringify(obj, null, 1), { flag: 'wx' });
  renameSync(tmp, path);
};

/**
 * Play every pair of a stage, writing evidence as it goes. Seams: `players` (the two player objects), `bgCheck`
 * (returns {ok, output}), `now`, `signal`, `nPairs` (tests only; production passes PAIRS[stage]). `header` is copied
 * into every sidecar and the record. Returns the record. Throws G3Void on a stop; the caller writes the VOID record.
 */
export async function runStage({ stage, outDir, players, nPairs, bgCheck, header, signal, now = () => performance.now(), startupMs }) {
  const gamesDir = join(outDir, 'games');
  mkdirSync(gamesDir);
  const progressPath = join(outDir, 'progress.jsonl');
  const completed = [];
  for (let pair = 0; pair < nPairs; pair++) {
    for (const game of [0, 1]) {
      const newIsRed = game === 0;
      const g = await playGame({ newPlayer: players.newHard, legacyPlayer: players.legacyHard, newIsRed, stage, pair, game, signal, now });
      const sidecar = {
        schema: SCHEMA,
        stage,
        pair_index: pair,
        game_in_pair: game,
        new_colour: newIsRed ? 'red' : 'black',
        red_player: newIsRed ? 'new_hard' : 'legacy_hard',
        black_player: newIsRed ? 'legacy_hard' : 'new_hard',
        new_model_id: players.newHard.modelId,
        legacy_model_id: players.legacyHard.modelId,
        match_seed: MATCH_SEED,
        moves: g.moves,
        per_move: g.perMove,
        result: g.result,
        termination: g.termination,
        ply_count: g.plyCount,
        new_score: g.newScore,
        move_ms_total: g.moveMsTotal,
        ...header,
      };
      writeJsonAtomic(join(gamesDir, sidecarName(pair, game)), sidecar);
      const bg = bgCheck();
      if (!bg.ok) throw new G3Void('background_compute', `after pair ${pair} game ${game}: ${bg.output.trim()}`);
      const line = { pair_index: pair, game_in_pair: game, ended_ms: now(), move_ms_total: g.moveMsTotal, ply_count: g.plyCount, bg_check: bg.output.trim() };
      appendFileSync(progressPath, `${JSON.stringify(line)}\n`);
      completed.push(`${pair}:${game}`);
      process.stdout.write(`${stage} pair ${pair} game ${game}: ${g.plyCount} plies, ${(g.moveMsTotal / 1000).toFixed(1)} s\n`);
    }
  }
  return { games_completed: completed.length, startup_ms: startupMs };
}

// ---------------------------------------------------------------------------------------------------------------
// Execution binding (card §7)
// ---------------------------------------------------------------------------------------------------------------

const IMPORT_RE = /\bfrom\s+['"](\.{1,2}\/[^'"]+)['"]|\bimport\s*\(?\s*['"](\.{1,2}\/[^'"]+)['"]/g;

/** Every repository file reached by relative imports from `entryRel`, sorted. Bare (package) imports are runtime. */
export function importClosure(entryRel, root = REPO_ROOT) {
  const seen = new Set();
  const stack = [normalize(entryRel)];
  while (stack.length) {
    const rel = stack.pop();
    if (seen.has(rel)) continue;
    seen.add(rel);
    const src = readFileSync(join(root, rel), 'utf8');
    for (const m of src.matchAll(IMPORT_RE)) {
      const spec = m[1] ?? m[2];
      stack.push(normalize(relative(root, join(root, dirname(rel), spec))));
    }
  }
  return [...seen].sort();
}

const hashed = (root, rels) => Object.fromEntries(rels.map((r) => [r, fileSha(root, r)]));

/** The binding: every byte the stages execute or judge, except the gate. Deterministic; no timestamps. */
export function computeBinding(root = REPO_ROOT) {
  const driver = importClosure(DRIVER_RELPATH, root).filter((r) => r !== GATE_RELPATH);
  const analyser = importClosure(ANALYSER_RELPATH, root).filter((r) => r !== GATE_RELPATH);
  const models = {};
  for (const [label, rel] of Object.entries(MODEL_MANIFESTS)) {
    const m = JSON.parse(readFileSync(join(root, rel), 'utf8'));
    models[label] = { model_id: m.model_id, manifest: rel, manifest_sha256: fileSha(root, rel) };
  }
  const body = {
    schema: BINDING_SCHEMA,
    match_seed: MATCH_SEED,
    pairs: { ...PAIRS },
    models,
    runtime: { node: process.version, files: hashed(root, ORT_FILES) },
    driver_files: hashed(root, driver),
    analyser_files: hashed(root, analyser),
    extra_files: hashed(root, EXTRA_FILES),
  };
  return { ...body, digest: sha256(JSON.stringify(body)) };
}

/** Differences between two bindings, as dotted paths. */
export function bindingDiff(a, b, path = '') {
  if (a === b) return [];
  if (typeof a !== 'object' || typeof b !== 'object' || a === null || b === null) return [path || '(root)'];
  const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
  return [...keys].flatMap((k) => bindingDiff(a[k], b[k], path ? `${path}.${k}` : k));
}

// ---------------------------------------------------------------------------------------------------------------
// Production entry
// ---------------------------------------------------------------------------------------------------------------

const git = (...args) => execFileSync('git', args, { cwd: REPO_ROOT }).toString();

function committedJson(rel) {
  let blob;
  try {
    blob = git('show', `HEAD:${rel}`);
  } catch {
    refuse('NOT_COMMITTED', `${rel} is not committed at HEAD`);
  }
  if (readFileSync(join(REPO_ROOT, rel), 'utf8') !== blob) refuse('NOT_COMMITTED', `${rel} differs from HEAD`);
  return JSON.parse(blob);
}

/** The preconditions of card §8, in order; any failure refuses before a model is read. */
export function preflight(stage, args, env = process.env) {
  if (!GATES[GATE_FOR[stage]]) refuse('GATE_CLOSED', `${GATE_FOR[stage]} is not open`);
  for (const k of ['MODEL_MANIFEST', 'HARD_MODEL_MANIFEST', 'HARD_MODE']) {
    if (env[k] !== undefined) refuse('OVERRIDE_SET', `${k} is set; the match uses the pinned defaults only`);
  }
  if (git('status', '--porcelain', '--untracked-files=no') !== '') refuse('DIRTY', 'tracked files differ from HEAD');
  const bound = committedJson(BINDING_RELPATH);
  const diff = bindingDiff(bound, computeBinding());
  if (diff.length) refuse('BINDING_MISMATCH', `differs from ${BINDING_RELPATH}: ${diff.slice(0, 10).join(', ')}`);
  const outDir = join(REPO_ROOT, outDirRel(stage));
  if (existsSync(outDir)) refuse('EXISTS', `${outDirRel(stage)} exists (one-shot)`);
  let killS = null;
  if (stage === 'match') {
    const timing = committedJson(TIMING_RELPATH);
    if (timing.verdict !== 'WITHIN_24H') refuse('TIMING', `the committed timing verdict is ${timing.verdict}`);
    if (!Number.isInteger(args.killS) || args.killS !== timing.kill_s) {
      refuse('TIMING', `--kill-s ${args.killS} is not the committed ceiling ${timing.kill_s}`);
    }
    killS = timing.kill_s;
  }
  if (!args.python || !existsSync(args.python)) refuse('USAGE', '--python must name an existing interpreter');
  return { bound, outDir, killS, head: git('rev-parse', 'HEAD').trim() };
}

/** Load both models exactly as server/index.js does, with no override reachable. */
export async function loadPlayers() {
  const legacy = await resolveModel({}, REPO_ROOT);
  const hard = await resolveHardModel({}, REPO_ROOT);
  if (legacy.manifest.model_id !== LEGACY_MODEL_ID) refuse('MODEL', `default model is ${legacy.manifest.model_id}`);
  if (hard.manifest.model_id !== NEW_MODEL_ID) refuse('MODEL', `hard model is ${hard.manifest.model_id}`);
  const legacyInf = new AlphaZeroInference(legacy.graphPath);
  await legacyInf.load();
  assertSessionContract(legacy.manifest, legacyInf.session, legacyInf.maxMoves);
  const hardBase = new AlphaZeroInference(hard.graphPath);
  await hardBase.load();
  assertSessionContract(hard.manifest, hardBase.session, hardBase.maxMoves);
  return {
    newHard: makeNewHardPlayer({ inference: new MirrorAveragingInference(hardBase), modelId: NEW_MODEL_ID }),
    legacyHard: makeLegacyHardPlayer({ inference: legacyInf, modelId: LEGACY_MODEL_ID }),
  };
}

/** The background-compute check (BOINC; ledger #101), run after every game. Exit 0 is the only pass. */
export function pythonBgCheck(python) {
  return () => {
    const r = spawnSync(python, ['-B', '-m', 'scripts.GPU.alphazero.background_compute', 'check'], {
      cwd: REPO_ROOT,
      env: { ...process.env, PYTHONPATH: '.' },
      encoding: 'utf8',
      timeout: 60_000,
    });
    return { ok: r.status === 0, output: `${r.stdout ?? ''}${r.stderr ?? ''}${r.error ? String(r.error) : ''}` };
  };
}

function parseArgs(rest) {
  const a = { python: null, killS: null };
  for (let i = 0; i < rest.length; i++) {
    if (rest[i] === '--python') a.python = rest[++i];
    else if (rest[i] === '--kill-s') a.killS = Number(rest[++i]);
    else return null;
  }
  return a;
}

async function runCli(stage, args) {
  let pre;
  try {
    pre = preflight(stage, args);
  } catch (err) {
    console.error(`REFUSED ${err.code ?? ''}: ${err.message}`);
    return EXIT.REFUSED;
  }
  mkdirSync(pre.outDir);
  const recordPath = join(pre.outDir, 'record.json');
  const controller = new AbortController();
  let signalled = null;
  for (const sig of ['SIGTERM', 'SIGINT', 'SIGHUP']) {
    process.on(sig, () => {
      signalled = sig;
      controller.abort();
    });
  }
  const base = {
    schema: RECORD_SCHEMA,
    stage,
    pairs: PAIRS[stage],
    head: pre.head,
    binding_digest: pre.bound.digest,
    kill_s: pre.killS,
    python: args.python,
    started_at: new Date().toISOString(),
  };
  let players;
  try {
    players = await loadPlayers();
  } catch (err) {
    writeJsonAtomic(recordPath, { ...base, status: 'REFUSED', refusal: { code: err.code ?? 'LOAD', message: err.message } });
    console.error(`REFUSED: ${err.message}`);
    return EXIT.REFUSED;
  }
  const startupMs = performance.now();
  const header = {
    binding_digest: pre.bound.digest,
    head: pre.head,
    node_version: process.version,
    ort_version: JSON.parse(readFileSync(join(REPO_ROOT, ORT_FILES[0]), 'utf8')).version,
  };
  try {
    const r = await runStage({
      stage,
      outDir: pre.outDir,
      players,
      nPairs: PAIRS[stage],
      bgCheck: pythonBgCheck(args.python),
      header,
      signal: controller.signal,
      startupMs,
    });
    writeJsonAtomic(recordPath, { ...base, status: 'COMPLETED', ...r, ended_ms: performance.now(), finished_at: new Date().toISOString() });
    return EXIT.COMPLETED;
  } catch (err) {
    const step = err instanceof G3Void ? err.step : signalled ? 'terminated' : 'error';
    writeJsonAtomic(recordPath, {
      ...base,
      status: 'VOID',
      void: { step, signal: signalled, message: String(err?.stack ?? err) },
      startup_ms: startupMs,
      ended_ms: performance.now(),
      finished_at: new Date().toISOString(),
    });
    console.error(`VOID at ${step}: ${err?.message ?? err}`);
    return EXIT.VOID;
  }
}

const isMain = process.argv[1] && import.meta.url === `file://${process.argv[1]}`;
if (isMain) {
  const [cmd, ...rest] = process.argv.slice(2);
  if (cmd === 'bind' && (rest.length === 0 || (rest.length === 1 && rest[0] === '--write'))) {
    const text = `${JSON.stringify(computeBinding(), null, 1)}\n`;
    if (rest[0] === '--write') writeFileSync(join(REPO_ROOT, BINDING_RELPATH), text);
    else process.stdout.write(text);
  } else if (cmd === 'run' && (rest[0] === 'smoke' || rest[0] === 'match') && parseArgs(rest.slice(1))) {
    process.exitCode = await runCli(rest[0], parseArgs(rest.slice(1)));
  } else {
    console.error('usage: g3_match.mjs bind [--write] | run smoke --python <py> | run match --python <py> --kill-s <s>');
    process.exitCode = EXIT.USAGE;
  }
}
