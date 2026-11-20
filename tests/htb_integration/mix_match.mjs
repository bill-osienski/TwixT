/**
 * H-MX match driver. Card: docs/superpowers/2026-11-18-htb-mix-distillation-card.md (revision 1), §9, §11, §13.
 *
 * H-OP's driver (tests/htb_integration/onpolicy_match.mjs) with H-MX's constants: the challenger (the exported pair in
 * --challenger-dir, qualified by the committed H-MX export record of EXPORT ATTEMPT 2, export amendment 1 §4; attempt
 * 1's VOID record is never read) against the current hard (78c16a31c3dba37f,
 * resolveHardModel), both as the product builds hard: MirrorAveragingInference(AlphaZeroInference) through protocolMove
 * (200 simulations, T 1.0 readout through ply 19, root halving from ply 20), each evaluation behind H-OP's non-finite
 * guard (imported unmodified). One Node process, serial.
 *
 * Pair p plays opening p of the committed H-MX openings file (its "match" list of 400, or "match_smoke" for the smoke):
 * game 0 with the challenger as Red, game 1 as Black. Seeds per move: sha256(`htb-mix/20261118/<match|match smoke>/pair
 * p/game g/ply k`), hex 0-11 search and 12-23 readout; only the side to move draws. The PRODUCT's game-end rule only
 * (Node TwixtState.isTerminal(), no cap). Every move records its budget (nSimulations) and the root visits spent.
 *
 * Usage: node tests/htb_integration/mix_match.mjs run <smoke|match> --challenger-dir <dir> --kill-s <s>
 * Exit (set through process.exitCode; never process.exit, ledger entry 64): 0 COMPLETED, 2 USAGE, 3 VOID, 4 REFUSED.
 */
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, join, resolve } from 'node:path';
import { performance } from 'node:perf_hooks';
import { fileURLToPath } from 'node:url';

import { TwixtState } from '../../server/gameLogic.js';
import { protocolMove } from '../../server/protocol_move.js';
import { resolvePolicy } from '../../server/readout_policy.js';
import { EXPORT_AMENDMENT_SHA256, EXPORT_ATTEMPT, EXPORT_OUTER_REL, WRAPPER_FILES, exportWrapperFailures } from './mix_match_analyse.mjs';
import { MatchRefusal, MatchStop, finiteGuard } from './onpolicy_match.mjs';

export { MatchRefusal, MatchStop, finiteGuard };

export const REPO_ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
export const NS = 'htb-mix/20261118';
export const CURRENT_MODEL_ID = '78c16a31c3dba37f';
export const OPENINGS_REL = 'docs/superpowers/evidence/htb-mix-openings/openings.json';
export const OPENINGS_SHA256 = 'f4ae110ff9e7547e413b15fff29ca0ff94d34c5f21d5d8b7da76af6577ee3563';
export const EXPORT_REL = 'docs/superpowers/evidence/htb-mix-export-2/record.json'; // export attempt 2 (amendment §4)
export const EXPORT_KIND = 'htb-mix-export-record/1';
export const PAIRS = { smoke: 2, match: 400 };
export const KILL_S = { smoke: 1800, match: 64800 };
export const SEED_STAGE = { smoke: 'match smoke', match: 'match' };
export const OPENING_LIST = { smoke: 'match_smoke', match: 'match' };
export const SIMS = 200;
export const RUN_REL = {
  smoke: 'docs/superpowers/evidence/htb-mix-match-smoke',
  match: 'docs/superpowers/evidence/htb-mix-match',
};
export const OUTER_REL = {
  smoke: 'docs/superpowers/evidence/htb-mix-match-smoke-run',
  match: 'docs/superpowers/evidence/htb-mix-match-run',
};
export const TIMING_REL = `${RUN_REL.smoke}/timing.json`;
export const GATES = { smoke: 'MIX_MATCH_SMOKE_AUTHORIZED', match: 'MIX_MATCH_AUTHORIZED' };
export const ORT_VERSION = '1.23.2';
export const GAME_SCHEMA = 'htb-mix-match-game/1';
export const RECORD_SCHEMA = 'htb-mix-match-record/1';

/** The onnxruntime-node the server's inference module actually resolves. */
export function ortVersion() {
  const req = createRequire(join(REPO_ROOT, 'server', 'inference.js'));
  return req('onnxruntime-node/package.json').version;
}

const gitAt = (root) => (...args) => spawnSync('git', args, { cwd: root, encoding: 'utf8' });
const git = gitAt(REPO_ROOT);

const sha256 = (s) => createHash('sha256').update(s).digest('hex');
const key = (m) => `${m[0]},${m[1]}`;

// ---------------------------------------------------------------------------------------------- openings and seeds

/** The committed openings file (built by htb_mix.py at build time), refused unless it is the pinned bytes. */
export function loadOpenings(root = REPO_ROOT) {
  const p = join(root, OPENINGS_REL);
  if (!existsSync(p)) throw new MatchRefusal(`${OPENINGS_REL} is missing`);
  const raw = readFileSync(p);
  if (sha256(raw) !== OPENINGS_SHA256) throw new MatchRefusal(`${OPENINGS_REL} is not the pinned file (sha256 ${OPENINGS_SHA256})`);
  return JSON.parse(raw.toString('utf8'));
}

/** Every game of a stage, in play order: pair p twice, the challenger Red in game 0 and Black in game 1. */
export function schedule(stage, openings = loadOpenings()) {
  const list = openings[OPENING_LIST[stage]];
  if (!Array.isArray(list) || list.length !== PAIRS[stage]) throw new MatchRefusal(`the ${OPENING_LIST[stage]} list does not hold ${PAIRS[stage]} openings`);
  const out = [];
  for (let p = 0; p < PAIRS[stage]; p++) {
    for (const g of [0, 1]) out.push({ pair: p, game: g, challengerColour: g === 0 ? 'red' : 'black', openingIndex: list[p].index, opening: list[p].moves });
  }
  return out;
}

/** The mover's seeds for one ply: two 48-bit integers, as in G3, the benchmark and H-OP. */
export function moveSeeds(stage, pair, game, ply) {
  const h = sha256(`${NS}/${SEED_STAGE[stage]}/pair ${pair}/game ${game}/ply ${ply}`);
  return { search: parseInt(h.slice(0, 12), 16), readout: parseInt(h.slice(12, 24), 16) };
}

// ------------------------------------------------------------------------------------------------------- players

/** A hard player: protocolMove with the stage's per-ply seeds; records the budget given and the root visits spent. */
export function makePlayer(who, rawInference, policy) {
  const inference = finiteGuard(rawInference, who);
  return {
    who,
    async move(state, { stage, pair, game }) {
      const seeds = moveSeeds(stage, pair, game, state.ply);
      const gameState = TwixtState.fromDict(state.toDict());
      const out = await protocolMove({ gameState, policy, inference, seeds });
      let visits = 0;
      for (const v of out.visitCounts.values()) visits += v;
      return { moveKey: out.moveKey, extra: { seeds, nSimulations: out.nSimulations, visits } };
    },
  };
}

function applyLegal(state, mv, who) {
  if (!state.legalMoves().some((m) => m[0] === mv[0] && m[1] === mv[1])) {
    throw new MatchStop('illegal', `${who} chose ${key(mv)}, illegal at ply ${state.ply}`);
  }
  return state.applyMove(mv);
}

/** One game under the product's game-end rule. `players` = { challenger, current }. */
export async function playGame({ stage, task, players }) {
  const t0 = performance.now();
  let state = new TwixtState();
  for (const mv of task.opening) state = applyLegal(state, mv, 'opening');
  const moves = task.opening.map((m) => [...m]);
  const perMove = [];
  while (!state.isTerminal()) {
    const ply = state.ply;
    const mover = state.toMove === task.challengerColour ? 'challenger' : 'current';
    const m0 = performance.now();
    const out = await players[mover].move(state, { stage, pair: task.pair, game: task.game });
    const ms = performance.now() - m0;
    const mv = out.moveKey.split(',').map(Number);
    state = applyLegal(state, mv, mover);
    moves.push(mv);
    perMove.push({ ply, mover, move: mv, ms, ...out.extra });
  }
  const winner = state.winner();
  return {
    schema: GAME_SCHEMA,
    stage,
    pair: task.pair,
    game: task.game,
    challenger_colour: task.challengerColour,
    opening_index: task.openingIndex,
    opening: task.opening,
    moves,
    plies: moves.length,
    result: winner ?? 'draw',
    challenger_score: winner === null ? 0.5 : winner === task.challengerColour ? 1 : 0,
    termination: winner === null ? 'no_legal_moves' : 'connection',
    wall_ms: performance.now() - t0,
    per_move: perMove,
  };
}

// --------------------------------------------------------------------------------------------------- admission

/**
 * The committed H-MX export record qualifies exactly the pair in `challengerDir` (card §8), with its wrapper's
 * evidence committed, clean and bound (H-OP's handoff repair). An empty list admits. `g` runs git at the root;
 * `fileSha` hashes a file (both injectable for the tests).
 */
export function exportRecordProblems(challengerDir, { root = REPO_ROOT, g = gitAt(root), fileSha = (p) => sha256(readFileSync(p)) } = {}) {
  const p = [];
  const recPath = join(root, EXPORT_REL);
  if (!existsSync(recPath)) return [`${EXPORT_REL} is missing`];
  if (g('ls-files', '--error-unmatch', '--', EXPORT_REL).status !== 0) p.push(`${EXPORT_REL} is not committed`);
  if (g('diff', '--quiet', 'HEAD', '--', EXPORT_REL).status !== 0) p.push(`${EXPORT_REL} differs from HEAD`);
  let rec;
  try {
    rec = JSON.parse(readFileSync(recPath, 'utf8'));
  } catch {
    return [...p, `${EXPORT_REL} is unparseable`];
  }
  const want = { kind: EXPORT_KIND, status: 'COMPLETED', verdict: 'PARITY_PASS', attempt: EXPORT_ATTEMPT, export_amendment_sha256: EXPORT_AMENDMENT_SHA256 };
  for (const [k, v] of Object.entries(want)) if (rec?.[k] !== v) p.push(`export record ${k} ${JSON.stringify(rec?.[k])}, expected ${JSON.stringify(v)}`);
  const c = rec?.challenger ?? {};
  for (const k of ['model_id', 'dir', 'graph_sha256', 'external_data_sha256', 'source_sha256']) if (typeof c[k] !== 'string' || !c[k]) p.push(`export record challenger.${k} missing`);
  for (const f of WRAPPER_FILES) {
    const rel = `${EXPORT_OUTER_REL}/${f}`;
    if (!existsSync(join(root, rel)) || g('ls-files', '--error-unmatch', '--', rel).status !== 0) p.push(`${rel} is not committed`);
  }
  const evDirs = [dirname(EXPORT_REL), EXPORT_OUTER_REL];
  if (g('diff', '--quiet', 'HEAD', '--', ...evDirs).status !== 0) p.push('the export evidence differs from HEAD');
  const others = g('ls-files', '--others', '--exclude-standard', '--', ...evDirs).stdout.trim();
  if (others) p.push(`untracked export evidence: ${others.split('\n').slice(0, 3).join(', ')}`);
  if (p.length) return p;
  let outer = null;
  try {
    outer = JSON.parse(readFileSync(join(root, EXPORT_OUTER_REL, 'outer.json'), 'utf8'));
  } catch {
    return [...p, `${EXPORT_OUTER_REL}/outer.json is unparseable`];
  }
  p.push(...exportWrapperFailures({ outer, pgid: readFileSync(join(root, EXPORT_OUTER_REL, 'pgid.txt'), 'utf8'), exportRecord: rec }));
  if (p.length) return p;
  if (!/^[0-9a-f]{16}$/.test(c.model_id)) p.push(`challenger model_id ${c.model_id} is not 16 hex`);
  if (c.model_id === CURRENT_MODEL_ID) p.push(`the challenger is the current hard ${CURRENT_MODEL_ID}`);
  if (!challengerDir || resolve(challengerDir) !== resolve(c.dir)) p.push(`--challenger-dir ${challengerDir} is not the export record's ${c.dir}`);
  const files = { 'model.onnx': c.graph_sha256, 'model.onnx.data': c.external_data_sha256 };
  for (const [f, want256] of Object.entries(files)) {
    const fp = join(c.dir, f);
    if (!existsSync(fp)) p.push(`${fp} is missing`);
    else if (fileSha(fp) !== want256) p.push(`${fp} sha256 is not the export record's`);
  }
  const mp = join(c.dir, 'manifest.json');
  if (!existsSync(mp)) p.push(`${mp} is missing`);
  else {
    let m;
    try {
      m = JSON.parse(readFileSync(mp, 'utf8'));
    } catch {
      m = null;
    }
    if (m?.model_id !== c.model_id) p.push(`the manifest's model_id ${m?.model_id} is not the export record's ${c.model_id}`);
    if (m?.graph?.sha256 !== c.graph_sha256 || m?.external_data?.sha256 !== c.external_data_sha256) p.push('the manifest does not name the exported pair');
  }
  return p;
}

/** Match admission (card §11): the smoke's evidence is committed and unchanged at HEAD, with nothing untracked. */
export function smokeEvidenceProblems(g = git) {
  const p = [];
  const required = ['record.json', 'timing.json', 'progress.jsonl'].map((f) => `${RUN_REL.smoke}/${f}`)
    .concat(['outer.json', 'pgid.txt', 'idle_check.txt', 'bg_check.txt'].map((f) => `${OUTER_REL.smoke}/${f}`));
  for (const f of required) if (g('ls-files', '--error-unmatch', '--', f).status !== 0) p.push(`${f} is not committed`);
  const games = g('ls-files', '--', `${RUN_REL.smoke}/games`).stdout.trim().split('\n').filter(Boolean);
  if (games.length !== 2 * PAIRS.smoke) p.push(`${games.length} committed smoke games, expected ${2 * PAIRS.smoke}`);
  if (g('diff', '--quiet', 'HEAD', '--', RUN_REL.smoke, OUTER_REL.smoke).status !== 0) p.push('smoke evidence differs from HEAD');
  const others = g('ls-files', '--others', '--exclude-standard', '--', RUN_REL.smoke, OUTER_REL.smoke).stdout.trim();
  if (others) p.push(`untracked smoke evidence: ${others.split('\n').slice(0, 3).join(', ')}`);
  return p;
}

// ---------------------------------------------------------------------------------------------------------- the run

function parseArgs(argv) {
  const [cmd, stage, ...rest] = argv;
  const opts = {};
  for (let i = 0; i < rest.length; i += 2) {
    if (!rest[i]?.startsWith('--') || rest[i + 1] === undefined) return null;
    opts[rest[i].slice(2)] = rest[i + 1];
  }
  return { cmd, stage, opts };
}

async function run(stage, opts) {
  const GATE = await import('./mix_authorization.mjs');
  if (GATE[GATES[stage]] !== true) throw new MatchRefusal(`${GATES[stage]} is closed`);
  for (const k of ['challenger-dir', 'kill-s']) if (!opts[k]) throw new MatchRefusal(`missing --${k}`);
  if (Number(opts['kill-s']) !== KILL_S[stage]) throw new MatchRefusal(`--kill-s ${opts['kill-s']} is not the card's ${KILL_S[stage]}`);
  const tasks = schedule(stage);
  const ep = exportRecordProblems(opts['challenger-dir']);
  if (ep.length) throw new MatchRefusal(`export record: ${ep.join('; ')}`);
  if (stage === 'match') {
    const sp = smokeEvidenceProblems();
    if (sp.length) throw new MatchRefusal(`smoke evidence: ${sp.join('; ')}`);
    const t = JSON.parse(readFileSync(join(REPO_ROOT, TIMING_REL), 'utf8'));
    if (t?.verdict !== 'LAUNCH_OK' || t.match_cap_s !== KILL_S.match) throw new MatchRefusal(`${TIMING_REL} is not LAUNCH_OK at the ${KILL_S.match} s cap`);
  }
  const out = join(REPO_ROOT, RUN_REL[stage]);
  if (existsSync(out)) throw new MatchRefusal(`${RUN_REL[stage]} exists (one-shot)`);
  const ort = ortVersion();
  if (ort !== ORT_VERSION) throw new MatchRefusal(`onnxruntime-node ${ort}, the card pins ${ORT_VERSION}`);
  const head = git('rev-parse', 'HEAD').stdout.trim();
  if (!/^[0-9a-f]{40}$/.test(head)) throw new MatchRefusal('no HEAD');
  const exp = JSON.parse(readFileSync(join(REPO_ROOT, EXPORT_REL), 'utf8')).challenger;

  const tStart = performance.now();
  const { resolveHardModel, loadManifest, validateArtifact } = await import('../../server/model_manifest.js');
  const { AlphaZeroInference } = await import('../../server/inference.js');
  const { MirrorAveragingInference } = await import('../../server/mirror_inference.js');
  const hard = await resolveHardModel({}, REPO_ROOT);
  if (hard.manifest.model_id !== CURRENT_MODEL_ID) throw new MatchRefusal(`hard model is ${hard.manifest.model_id}, expected ${CURRENT_MODEL_ID}`);
  const cManifest = await loadManifest(join(exp.dir, 'manifest.json'));
  const cArt = await validateArtifact(cManifest, exp.dir); // the product's own validation of a pair
  if (cManifest.model_id !== exp.model_id) throw new MatchRefusal(`challenger manifest ${cManifest.model_id} is not ${exp.model_id}`);
  const policy = resolvePolicy({ difficulty: 'hard', hardMode: 'protocol' });
  if (policy.nSims !== SIMS || policy.moveTemp !== 1.0 || policy.search !== 'protocol') throw new MatchRefusal(`unexpected policy ${JSON.stringify(policy)}`);

  const started = new Date().toISOString();
  const loaded = {};
  for (const [who, graphPath] of [['challenger', cArt.graphPath], ['current', hard.graphPath]]) {
    const base = new AlphaZeroInference(graphPath);
    await base.load();
    loaded[who] = { inference: new MirrorAveragingInference(base), onnxSha: sha256(readFileSync(graphPath)) };
  }
  if (loaded.challenger.onnxSha !== exp.graph_sha256) throw new MatchStop('challenger', 'the loaded challenger graph is not the export record\'s');
  mkdirSync(join(out, 'games'), { recursive: true });
  const startupSeconds = (performance.now() - tStart) / 1000;

  const players = {
    challenger: makePlayer('challenger', loaded.challenger.inference, policy),
    current: makePlayer('current', loaded.current.inference, policy),
  };
  const searchSeconds = { challenger: 0, current: 0 };
  const progress = join(out, 'progress.jsonl');
  let n = 0;
  for (const task of tasks) {
    const rec = await playGame({ stage, task, players });
    const name = `pair_${String(task.pair).padStart(4, '0')}_game_${task.game}.json`;
    writeFileSync(join(out, 'games', name), `${JSON.stringify(rec)}\n`, { flag: 'wx' });
    for (const pm of rec.per_move) searchSeconds[pm.mover] += pm.ms / 1000;
    n += 1;
    // Progress carries no outcome: games done and plies only.
    appendFileSync(progress, `${JSON.stringify({ t: new Date().toISOString(), pair: task.pair, game: task.game, plies: rec.plies, done: n, of: tasks.length })}\n`);
  }
  const record = {
    schema: RECORD_SCHEMA,
    written_by: 'script',
    status: 'COMPLETED',
    stage,
    started,
    ended: new Date().toISOString(),
    games: n,
    pairs: PAIRS[stage],
    startup_seconds: startupSeconds,
    search_seconds: searchSeconds,
    total_seconds: (performance.now() - tStart) / 1000,
    challenger: { model_id: exp.model_id, dir: exp.dir, graph_sha256: exp.graph_sha256, external_data_sha256: exp.external_data_sha256, source_sha256: exp.source_sha256, onnx_sha256: loaded.challenger.onnxSha },
    current: { model_id: CURRENT_MODEL_ID, onnx_sha256: loaded.current.onnxSha },
    inference: 'MirrorAveragingInference(AlphaZeroInference), both',
    policy,
    openings_sha256: OPENINGS_SHA256,
    kill_s: KILL_S[stage],
    head,
    pid: process.pid,
    runtime: { node: process.version, onnxruntime_node: ort },
  };
  writeFileSync(join(out, 'record.json'), `${JSON.stringify(record, null, 1)}\n`, { flag: 'wx' });
}

async function main() {
  const a = parseArgs(process.argv.slice(2));
  if (!a || a.cmd !== 'run' || !['smoke', 'match'].includes(a.stage)) {
    console.error('usage: node tests/htb_integration/mix_match.mjs run <smoke|match> --challenger-dir <dir> --kill-s <s>');
    return 2;
  }
  try {
    await run(a.stage, a.opts);
    return 0;
  } catch (err) {
    if (err instanceof MatchRefusal) {
      console.error(`REFUSED: ${err.message}`);
      return 4;
    }
    const out = join(REPO_ROOT, RUN_REL[a.stage]);
    const step = err instanceof MatchStop ? err.step : 'crashed';
    console.error(err instanceof MatchStop ? `STOP at ${err.step}: ${err.message}` : (err?.stack ?? String(err)));
    if (existsSync(out) && !existsSync(join(out, 'record.json'))) {
      writeFileSync(join(out, 'record.json'), `${JSON.stringify({ schema: RECORD_SCHEMA, written_by: 'script', status: 'VOID', stage: a.stage, void: { step, detail: err?.message ?? String(err) } }, null, 1)}\n`, { flag: 'wx' });
    }
    return err instanceof MatchStop ? 3 : 1;
  }
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  // ledger entry 64: never process.exit() with ONNX Runtime loaded.
  main().then((code) => {
    process.exitCode = code;
  });
}
