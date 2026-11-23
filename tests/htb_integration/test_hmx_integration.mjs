/**
 * H-MX integration build: model-free tests (proposal docs/superpowers/2026-11-22-htb-mix-integration-proposal.md,
 * revision 1, §4: B2, B5 and the check tooling). No model loads, no server starts and no gate opens here.
 *
 * - the check gate is a JavaScript gate and is CLOSED (the Python gate inventory scans only Python files);
 * - the driver refuses while it is closed, writing nothing;
 * - B5: the seed preload serves its queue in order to `import { randomInt } from 'node:crypto'`, logs it, fails
 *   closed, passes other calls through, refuses without its environment, and drives the real protocol_move.js exactly
 *   as explicit seeds do; no product file imports it;
 * - the fixed samples, the comparators, the suite parser and the record admission, each against a clean control;
 * - B2: production server code differs from REF_COMMIT by exactly the planned HARD_MODEL_ID hunk (or not at all,
 *   before the swap and after its revert), server/test_hard_model.js is the one server test file allowed to change,
 *   every changed path is on the build's allowlist, and the served constants are unchanged by value.
 */
import assert from 'node:assert/strict';
import { execFileSync, spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { pathToFileURL } from 'node:url';

import { HARD_MODEL_ID } from '../../server/model_manifest.js';
import { DIFFICULTY_TABLE, LEGACY_HARD, PROTOCOL_HARD } from '../../server/readout_policy.js';
import { HALVING } from '../../server/root_halving.js';
import * as GATES from './hmx_integration_authorization.mjs';
import * as D from './hmx_integration_check.mjs';

const ROOT = D.REPO_ROOT;
const PRELOAD = join(ROOT, D.PRELOAD_REL);
const sha256 = (b) => createHash('sha256').update(b).digest('hex');
const git = (...a) => execFileSync('git', a, { cwd: ROOT, encoding: 'utf8' });
const tmp = () => mkdtempSync(join(tmpdir(), 'twixt-hmx-'));

// --------------------------------------------------------------------------------------------- the gate

test('the check gate is a closed JavaScript gate, alone in its file', () => {
  assert.equal(GATES.HMX_INTEGRATION_CHECK_AUTHORIZED, false);
  assert.deepEqual(Object.keys(GATES), [D.GATE]);
  const text = readFileSync(join(ROOT, D.GATE_REL), 'utf8');
  assert.deepEqual([...text.matchAll(/^export const (\w+) = (\w+);$/gm)].map((m) => [m[1], m[2]]), [[D.GATE, 'false']]);
});

test('the driver refuses while the gate is closed, and on bad usage, writing nothing', () => {
  const run = (...args) => spawnSync(process.execPath, [join(ROOT, 'tests/htb_integration/hmx_integration_check.mjs'), ...args], { encoding: 'utf8', cwd: ROOT });
  const r = run('run', 'check', '--ref-root', '/nonexistent', '--revert-root', '/nonexistent', '--kill-s', String(D.KILL_S));
  assert.equal(r.status, 4, r.stderr);
  assert.match(r.stderr, /REFUSED: HMX_INTEGRATION_CHECK_AUTHORIZED is closed/);
  assert.equal(run('run', 'other').status, 2);
  assert.equal(run().status, 2);
  assert.ok(!existsSync(join(ROOT, D.RUN_REL)));
});

// --------------------------------------------------------------------------------------------- B5: the preload

function toy(dir, body) {
  const main = join(dir, 'main.mjs');
  writeFileSync(join(dir, 'mod.mjs'), "import { randomInt } from 'node:crypto';\nexport const draw = () => [randomInt(0, 2 ** 48 - 1), randomInt(0, 2 ** 48 - 1)];\n");
  writeFileSync(main, body);
  return main;
}
const withQueue = (dir, queue) => {
  const f = join(dir, 'q.json');
  writeFileSync(f, JSON.stringify(queue));
  return f;
};
const node = (args, env) => {
  const e = { ...process.env, ...env };
  for (const k of ['HMX_SEED_QUEUE', 'NODE_OPTIONS']) if (!(k in env)) delete e[k];
  return spawnSync(process.execPath, args, { encoding: 'utf8', env: e });
};

test('B5: the preload serves the queue in order to an ESM named import, logs each value, and passes other calls through', () => {
  const d = tmp();
  try {
    const main = toy(d, "import { randomInt } from 'node:crypto';\nimport { draw } from './mod.mjs';\nconst a = draw(); const b = draw(); const c = randomInt(0, 10);\nconsole.log(JSON.stringify({ a, b, c }));\n");
    const q = [11, 22, 33, 44];
    const r = node(['--import', PRELOAD, main], { HMX_SEED_QUEUE: withQueue(d, q) });
    assert.equal(r.status, 0, r.stderr);
    const o = JSON.parse(r.stdout);
    assert.deepEqual([o.a, o.b], [[11, 22], [33, 44]]);
    assert.ok(Number.isInteger(o.c) && o.c >= 0 && o.c < 10);
    assert.deepEqual(D.seedLogProblems(r.stderr, q), []);
    // Control: without the preload the same program draws fresh values.
    const c = node([main], {});
    assert.equal(c.status, 0, c.stderr);
    assert.notDeepEqual(JSON.parse(c.stdout).a, [11, 22]);
    assert.ok(!/HMX_SEED/.test(c.stderr));
  } finally {
    rmSync(d, { recursive: true });
  }
});

test('B5: the preload fails closed: an exhausted queue throws, and it refuses to load without a valid queue', () => {
  const d = tmp();
  try {
    const main = toy(d, "import { draw } from './mod.mjs';\nconsole.log(JSON.stringify(draw())); console.log(JSON.stringify(draw()));\n");
    const r = node(['--import', PRELOAD, main], { HMX_SEED_QUEUE: withQueue(d, [1, 2, 3]) });
    assert.notEqual(r.status, 0);
    assert.match(r.stderr, /seed queue exhausted after 3 values/);
    assert.equal(r.stdout.trim().split('\n').length, 1);
    const unset = node(['--import', PRELOAD, main], {});
    assert.notEqual(unset.status, 0);
    assert.match(unset.stderr, /HMX_SEED_QUEUE is not set; refusing to load/);
    for (const bad of [[-1], [1.5], ['7'], [2 ** 48], { a: 1 }]) {
      const b = node(['--import', PRELOAD, main], { HMX_SEED_QUEUE: withQueue(d, bad) });
      assert.notEqual(b.status, 0, JSON.stringify(bad));
      assert.match(b.stderr, /is not an array of integers/);
    }
  } finally {
    rmSync(d, { recursive: true });
  }
});

test('B5: under the preload the real protocolMove, unseeded, plays exactly what explicit seeds play (fake evaluator)', () => {
  const d = tmp();
  try {
    const corpus = JSON.parse(readFileSync(join(ROOT, 'tests/parity/corpus.json'), 'utf8'));
    const all = [...corpus.primary, ...corpus.edge];
    const picks = [all.find((p) => p.ply >= 4 && p.ply < 20), all.find((p) => p.ply >= 20 && p.ply < 60)].map((p) => p.moves);
    const seeds = [{ search: 123456789, readout: 987654321 }, { search: 2 ** 47 + 5, readout: 77 }];
    const u = (rel) => pathToFileURL(join(ROOT, rel)).href;
    const main = join(d, 'pm.mjs');
    writeFileSync(main, [
      `import { TwixtState } from '${u('server/gameLogic.js')}';`,
      `import { protocolMove } from '${u('server/protocol_move.js')}';`,
      `import { resolvePolicy } from '${u('server/readout_policy.js')}';`,
      `import { MirrorAveragingInference } from '${u('server/mirror_inference.js')}';`,
      `import { fakeInference } from '${u('tests/fixtures/htb_integration/fake_evaluator.mjs')}';`,
      "const policy = resolvePolicy({ difficulty: 'hard', hardMode: 'protocol' });",
      `const picks = ${JSON.stringify(picks)}; const seeds = ${JSON.stringify(seeds)};`,
      'const view = (o) => ({ moveKey: o.moveKey, visits: [...o.visitCounts.entries()].sort(), rootValue: o.rootValue, valueKind: o.valueKind });',
      'const out = [];',
      'for (const [i, mv] of picks.entries()) {',
      '  const inf = new MirrorAveragingInference(fakeInference());',
      '  const fresh = view(await protocolMove({ gameState: TwixtState.fromMoves(mv), policy, inference: inf }));',
      '  const fixed = view(await protocolMove({ gameState: TwixtState.fromMoves(mv), policy, inference: inf, seeds: seeds[i] }));',
      '  out.push({ fresh, fixed });',
      '}',
      'console.log(JSON.stringify(out));',
    ].join('\n'));
    const q = D.seedQueue(seeds);
    const r = node(['--import', PRELOAD, main], { HMX_SEED_QUEUE: withQueue(d, q) });
    assert.equal(r.status, 0, r.stderr);
    const out = JSON.parse(r.stdout);
    for (const o of out) assert.deepEqual(o.fresh, o.fixed);
    assert.deepEqual(D.seedLogProblems(r.stderr, q), []); // explicit seeds consume nothing
  } finally {
    rmSync(d, { recursive: true });
  }
});

test('B5: no product file imports the preload or reads its queue variable', () => {
  const product = git('ls-files', 'server', 'assets', 'scripts/startServer.js', 'TwixT.html', 'Replay.html', 'package.json')
    .trim().split('\n').filter((f) => f && !/^server\/test_/.test(f) && !/\.(onnx|data|png|jpg|glb|bin)$/.test(f));
  assert.ok(product.includes('server/index.js') && product.includes('server/protocol_move.js'));
  for (const f of product) {
    const text = readFileSync(join(ROOT, f), 'utf8');
    assert.ok(!/hmx_seed_preload|HMX_SEED_QUEUE|hmx_integration/.test(text), f);
  }
  assert.ok(!JSON.stringify(JSON.parse(readFileSync(join(ROOT, 'package.json'), 'utf8')).scripts).includes('--import'));
});

// --------------------------------------------------------------------------------------------- the fixed samples

test('the replay sample is fixed: 41 games, 845 challenger rows, 844 current rows; 142 capture positions', () => {
  const games = D.selectGames();
  assert.equal(games.length, D.EXPECTED.games);
  assert.equal(games.filter((g) => g === D.REPLAY_EXTRA_GAME).length, 1);
  assert.ok(games.every((g) => g === D.REPLAY_EXTRA_GAME || Number(g.slice(5, 9)) % 20 === 0));
  assert.equal(sha256(readFileSync(join(ROOT, D.MATCH_REL, 'record.json'))), D.MATCH_RECORD_SHA256);
  const ch = D.replayRows('challenger');
  const cu = D.replayRows('current');
  assert.equal(ch.length, D.EXPECTED.challenger);
  assert.equal(cu.length, D.EXPECTED.current);
  for (const r of [...ch, ...cu]) {
    assert.equal(r.nSimulations, 200);
    assert.equal(r.visits, 200);
    assert.ok(Number.isSafeInteger(r.seeds.search) && Number.isSafeInteger(r.seeds.readout));
  }
  assert.equal(Math.max(...ch.map((r) => r.ply)) >= 500, true); // the long game reaches the nearly full board
  assert.equal(D.capturePositions().length, D.EXPECTED.positions);
});

test('capture seeds are the namespaced sha256 prefixes, queued search then readout', () => {
  const id = 'p_0001';
  const h = (k) => parseInt(sha256(`hmx-integration/20261122/${id}/${k}`).slice(0, 12), 16);
  assert.deepEqual(D.positionSeeds(id), { search: h('search'), readout: h('readout') });
  assert.deepEqual(D.seedQueue([{ search: 1, readout: 2 }, { search: 3, readout: 4 }]), [1, 2, 3, 4]);
});

test('the seed log check: a clean consumption passes; a missing, extra or reordered value fails', () => {
  const q = [5, 6, 7, 8];
  const log = (vals) => vals.map((v, i) => `HMX_SEED ${i} ${v}`).join('\n');
  assert.deepEqual(D.seedLogProblems(`noise\n${log(q)}\nmore`, q), []);
  assert.ok(D.seedLogProblems(log([5, 6, 7]), q).length > 0);
  assert.ok(D.seedLogProblems(log([5, 6, 7, 8, 9]), q).length > 0);
  assert.ok(D.seedLogProblems(log([5, 7, 6, 8]), q).length > 0);
});

// --------------------------------------------------------------------------------------------- the comparators

function fakeCapture(hardId) {
  const positions = Array.from({ length: D.EXPECTED.positions }, (_, i) => {
    const plain = (n) => ({ type: 'bestmove', keys: ['a', 'b'], visits: { '1,1': n }, rootValue: 0.1 * (i % 7), valueRed: -0.2, nSimulations: n });
    return {
      id: `p${i}`,
      hard: { type: 'bestmove', keys: ['k'], move: { row: i % 24, col: 3 }, visits: { '1,1': 120, '2,2': 80 }, rootValue: 0.3, moveValue: 0.25,
        moveVisits: 120, valueRed: 0.25, valueKind: 'played_move_mean', nSimulations: 200, search: 'protocol', modelId: hardId },
      easy: plain(100),
      medium: plain(400),
    };
  });
  return { refused: false, positions };
}

test('compareCaptures: exact and served rules pass their clean controls and fail each tamper', () => {
  const ref = fakeCapture(D.CURRENT_ID);
  assert.equal(D.compareCaptures(ref, fakeCapture(D.CURRENT_ID), 'exact').verdict, 'PASS');
  assert.equal(D.compareCaptures(ref, fakeCapture(D.NEW_ID), 'served', D.NEW_ID).verdict, 'PASS');
  const tamper = (f) => {
    const s = fakeCapture(D.CURRENT_ID);
    f(s.positions[77]);
    return s;
  };
  for (const f of [(p) => (p.easy.visits['1,1'] = 99), (p) => (p.medium.rootValue += 1e-12), (p) => (p.hard.moveValue = 0.26),
    (p) => (p.hard.move = { row: 0, col: 0 }), (p) => (p.hard.keys = ['k', 'x'])]) {
    assert.equal(D.compareCaptures(ref, tamper(f), 'exact').verdict, 'FAIL');
  }
  assert.equal(D.compareCaptures(ref, fakeCapture(D.CURRENT_ID), 'served', D.NEW_ID).verdict, 'FAIL');
  const easyMoved = fakeCapture(D.NEW_ID);
  easyMoved.positions[3].easy.valueRed = 0.5;
  assert.equal(D.compareCaptures(ref, easyMoved, 'served', D.NEW_ID).verdict, 'FAIL');
  const short = fakeCapture(D.CURRENT_ID);
  short.positions.pop();
  assert.equal(D.compareCaptures(ref, short, 'exact').verdict, 'FAIL');
  assert.equal(D.compareCaptures(ref, { refused: true }, 'exact').verdict, 'FAIL');
});

test('compareReplay: exact moves and budget; REST may omit nSimulations; each tamper fails', () => {
  const rows = [{ game: 'g', ply: 4, move: [3, 4], nSimulations: 200, visits: 200 }, { game: 'g', ply: 6, move: [5, 6], nSimulations: 200, visits: 200 }];
  const ok = () => rows.map((r) => ({ move: [...r.move], visitsTotal: 200, nSimulations: 200, search: 'protocol', modelId: D.NEW_ID }));
  assert.equal(D.compareReplay(rows, ok(), D.NEW_ID).verdict, 'PASS');
  assert.equal(D.compareReplay(rows, ok().map(({ nSimulations, ...a }) => a), D.NEW_ID).verdict, 'PASS');
  for (const f of [(a) => (a[1].move = [5, 7]), (a) => (a[0].visitsTotal = 199), (a) => (a[0].modelId = D.CURRENT_ID), (a) => (a[0].search = 'plain'),
    (a) => (a[1].nSimulations = 100), (a) => a.pop()]) {
    const a = ok();
    f(a);
    assert.equal(D.compareReplay(rows, a, D.NEW_ID).verdict, 'FAIL');
  }
});

test('suiteVerdict: exit 0 passes; the revert tree accepts exactly the linked-worktree set and nothing else', () => {
  const out = (names) => `ℹ tests 91\nℹ pass ${91 - names.length}\nℹ fail ${names.length}\nℹ cancelled 0\nℹ skipped 0\n\n✖ failing tests:\n\n${names.map((n) => `test at x.mjs:1:1\n✖ ${n} (1.5ms)\n  AssertionError`).join('\n\n')}\n`;
  const L = D.LINKED_WORKTREE_FAILURES;
  assert.equal(D.suiteVerdict({ label: 'integration', code: 0, text: out([]) }).verdict, 'PASS');
  assert.equal(D.suiteVerdict({ label: 'integration', code: 1, text: out([...L]) }).verdict, 'FAIL');
  assert.equal(D.suiteVerdict({ label: 'integration', code: 1, text: out([...L]), allowedFailures: L }).verdict, 'PASS');
  assert.equal(D.suiteVerdict({ label: 'integration', code: 1, text: out([...L, 'another']), allowedFailures: L }).verdict, 'FAIL');
  assert.equal(D.suiteVerdict({ label: 'integration', code: 1, text: out(L.slice(1)), allowedFailures: L }).verdict, 'FAIL');
  assert.equal(D.suiteVerdict({ label: 'wrapper', code: 0, text: '9 passed in 1.4s' }).verdict, 'PASS');
  assert.equal(D.suiteVerdict({ label: 'wrapper', code: 1, text: '1 failed, 8 passed in 1.4s' }).verdict, 'FAIL');
  assert.equal(D.suiteVerdict({ label: 'lint', code: 1, text: 'error' }).verdict, 'FAIL');
});

test('recordProblems: a consistent record admits; a missing check or an unfounded verdict does not', () => {
  const rec = { schema: D.RECORD_SCHEMA, status: 'COMPLETED', verdict: 'PASS', kill_s: D.KILL_S, staged: { hard_model_id: D.NEW_ID },
    checks: Object.fromEntries(D.CHECK_IDS.map((id) => [id, { verdict: 'PASS' }])) };
  assert.deepEqual(D.recordProblems(rec), []);
  const failed = structuredClone(rec);
  failed.checks.R2.verdict = 'FAIL';
  assert.ok(D.recordProblems(failed).length > 0);
  failed.verdict = 'FAIL';
  assert.deepEqual(D.recordProblems(failed), []);
  const missing = structuredClone(rec);
  delete missing.checks.A1;
  assert.ok(D.recordProblems(missing).length > 0);
});

// --------------------------------------------------------------------------------------------- B2: the diff allowlist

/** The one planned production change (proposal §3), as `git diff -U0` prints its +/- lines. */
const SWAP_HUNK = [
  "- * The validated H-TB student's export, pinned after its export parity passed",
  '- * (V2-R, ledger Post-closure entry 58, under the approved amendment',
  '- * docs/superpowers/2026-11-09-htb-student-parity-amendment.md). Changing it is a',
  '- * tracked, reviewable diff, as for DEFAULT_MODEL_ID. HARD_MODE=legacy never',
  '- * reads it.',
  "+ * The H-MX challenger's export, pinned after its export parity passed (ledger",
  '+ * Post-closure entry 76) and its 400-pair match against 78c16a31c3dba37f passed',
  '+ * (entry 78), under docs/superpowers/2026-11-22-htb-mix-integration-proposal.md.',
  '+ * Rollback pin: 78c16a31c3dba37f, by a restart with',
  '+ * HARD_MODEL_MANIFEST=models/78c16a31c3dba37f/manifest.json or a revert of this',
  '+ * line. Changing it is a tracked, reviewable diff, as for DEFAULT_MODEL_ID.',
  '+ * HARD_MODE=legacy never reads it.',
  "-export const HARD_MODEL_ID = '78c16a31c3dba37f';",
  "+export const HARD_MODEL_ID = '627c8d774df838a8';",
];

/** Paths the build (and its later gated-run commits) may change relative to REF_COMMIT. */
const ALLOWED = new Set([
  'models/627c8d774df838a8/model.onnx', 'models/627c8d774df838a8/model.onnx.data', 'models/627c8d774df838a8/manifest.json',
  'server/model_manifest.js', 'server/test_hard_model.js',
  'tests/htb_integration/test_g3.mjs', 'tests/mcts_golden/cases.mjs', 'tests/mcts_golden/test_capture.mjs', 'tests/mcts_golden/test_falsify.mjs',
  'tests/htb_integration/hmx_integration_authorization.mjs', 'tests/htb_integration/hmx_integration_check.mjs',
  'tests/htb_integration/hmx_seed_preload.mjs', 'tests/htb_integration/test_hmx_integration.mjs',
  'tests/hmx_integration_wrapper_spec.py', 'tests/test_hmx_integration_wrapper.py',
]);

test('B2: production server code differs from REF_COMMIT by exactly the planned hunk, or not at all', () => {
  const production = readdirSync(join(ROOT, 'server')).filter((f) => f.endsWith('.js') && !f.startsWith('test_')).map((f) => `server/${f}`);
  assert.ok(production.includes('server/index.js') && production.includes('server/model_manifest.js'));
  const lines = git('diff', '-U0', D.REF_COMMIT, 'HEAD', '--', ...production).split('\n').filter((l) => /^[-+]/.test(l) && !/^(---|\+\+\+) /.test(l));
  const changed = git('diff', '--name-only', D.REF_COMMIT, 'HEAD', '--', ...production).trim().split('\n').filter(Boolean);
  const mm = sha256(readFileSync(join(ROOT, 'server/model_manifest.js')));
  if (HARD_MODEL_ID === D.CURRENT_ID) {
    // before the swap, and after its revert
    assert.deepEqual(changed, []);
    assert.equal(mm, D.MODEL_MANIFEST_JS_SHA256.old);
  } else {
    assert.equal(HARD_MODEL_ID, D.NEW_ID);
    assert.deepEqual(changed, ['server/model_manifest.js']);
    assert.deepEqual(lines, SWAP_HUNK);
    assert.equal(mm, D.MODEL_MANIFEST_JS_SHA256.new);
  }
});

test('B2: every changed path is on the allowlist; server/test_hard_model.js is the one server test allowed to change', () => {
  const changed = git('diff', '--name-only', D.REF_COMMIT, 'HEAD').trim().split('\n').filter(Boolean);
  const off = changed.filter((f) => !ALLOWED.has(f) && !f.startsWith('docs/'));
  assert.deepEqual(off, []);
  const serverTests = changed.filter((f) => /^server\/test_/.test(f));
  assert.ok(serverTests.every((f) => f === 'server/test_hard_model.js'), JSON.stringify(serverTests));
  for (const id of ['78c16a31c3dba37f', 'c34b7ff3297c785a', '1d64027db521a50f']) {
    assert.equal(git('diff', '--name-only', D.REF_COMMIT, 'HEAD', '--', `models/${id}`).trim(), '', id);
  }
});

test('B2: the served constants are unchanged by value', () => {
  assert.deepEqual({ ...PROTOCOL_HARD }, { nSims: 200, search: 'protocol', moveTemp: 1.0 });
  assert.deepEqual({ ...LEGACY_HARD }, { nSims: 800, search: 'plain', moveTemp: 0 });
  assert.deepEqual(DIFFICULTY_TABLE.easy, { nSims: 100, search: 'plain', moveTemp: 1.0 });
  assert.deepEqual(DIFFICULTY_TABLE.medium, { nSims: 400, search: 'plain', moveTemp: 0.5 });
  assert.equal(DIFFICULTY_TABLE.hard, PROTOCOL_HARD);
  assert.deepEqual({ ...HALVING }, { FROM_PLY: 20, CANDIDATES: 16, C_VISIT: 50.0, C_SCALE: 1.0, PRIOR_FLOOR: 1e-12, BOARD: 24 });
});
