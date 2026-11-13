/**
 * Diagnostic O, the FROZEN analysis (plan docs/superpowers/2026-11-12-htb-diagnostic-o-plan.md §5–§6). Its sha256 is
 * recorded in the plan before either model is loaded. It imports nothing from the driver and reads only:
 *   <run>/student.jsonl, <run>/teacher.json and docs/superpowers/evidence/htb-diag-o-plan/race_groups.json.
 *
 * Every value is from Black's side (the student's colour in these games). Game-level comparisons only; the unit is
 * the game. Losses (L) are compared with wins (W); draws (D) are shown separately and never pooled.
 *
 * Usage: node tests/htb_integration/diag_o_analyse.mjs analyse <run_dir>   (JSON on stdout)
 */
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const GROUPS_REL = 'docs/superpowers/evidence/htb-diag-o-plan/race_groups.json';
export const PLIES = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19];
export const LOW_SHARE = 0.1;
export const BOOT_B = 10000;
export const BOOT_SEED = 20261112;
export const SIGNAL = Object.freeze({ MIN_LOSSES: 8, RATIO: 2, AUC_TEACHER_MIN: 0.75, AUC_STUDENT_MAX: 0.6 });

/** mulberry32: a small seeded PRNG so the bootstrap is reproducible. */
export function rng32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export const mean = (xs) => {
  const v = xs.filter((x) => x !== null && Number.isFinite(x));
  return v.length ? v.reduce((s, x) => s + x, 0) / v.length : null;
};

/** AUC for "a lower value marks a loss": P(v_L < v_W) + 0.5 P(tie). */
export function aucLowerIsLoss(lossVals, winVals) {
  let s = 0;
  let n = 0;
  for (const a of lossVals) for (const b of winVals) {
    if (a === null || b === null) continue;
    s += a < b ? 1 : a === b ? 0.5 : 0;
    n += 1;
  }
  return n ? s / n : null;
}

function quantile(sorted, q) {
  if (!sorted.length) return null;
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

/** Stratified bootstrap over games: resample L and W separately, B times; 95% percentile interval of stat(L*, W*). */
export function bootstrap(L, W, stat, { B = BOOT_B, seed = BOOT_SEED } = {}) {
  const r = rng32(seed);
  const draw = (xs) => xs.map(() => xs[Math.floor(r() * xs.length)]);
  const vals = [];
  for (let b = 0; b < B; b++) {
    const v = stat(draw(L), draw(W));
    if (v !== null && Number.isFinite(v)) vals.push(v);
  }
  vals.sort((a, b) => a - b);
  return { lo: quantile(vals, 0.025), hi: quantile(vals, 0.975), n: vals.length };
}

/** Wilson 95% interval for k of n. */
export function wilson(k, n) {
  if (n === 0) return { lo: null, hi: null };
  const z = 1.959963984540054;
  const p = k / n;
  const d = 1 + (z * z) / n;
  const c = (p + (z * z) / (2 * n)) / d;
  const h = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / d;
  return { lo: c - h, hi: c + h };
}

/** Per-game metrics from the joined positions of one game. */
export function gameMetrics(pos) {
  const byPly = Object.fromEntries(pos.map((p) => [p.ply, p]));
  const alt = (leaderKey, searchKey) => {
    const rows = pos.filter((p) => p[leaderKey] !== p.played);
    const d = (f) => mean(rows.map(f));
    return {
      n: rows.length,
      student_raw: d((p) => p.student_raw.after_value_black[p[leaderKey]] - p.student_raw.after_value_black[p.played]),
      teacher_raw: d((p) => p.teacher.after_value_black[p[leaderKey]] - p.teacher.after_value_black[p.played]),
      search: d((p) => {
        const s = p[searchKey].move_value_black;
        return s.leader !== null && s.played !== null ? s.leader - s.played : null;
      }),
      teacher_prefers_alt: rows.filter((p) => p.teacher.after_value_black[p[leaderKey]] > p.teacher.after_value_black[p.played]).length,
      student_prefers_alt: rows.filter((p) => p.student_raw.after_value_black[p[leaderKey]] > p.student_raw.after_value_black[p.played]).length,
    };
  };
  const vals = (ply) => {
    const p = byPly[ply];
    return {
      student_raw: p.student_raw.value_black,
      teacher_raw: p.teacher.value_black,
      search200_root: p.search200.root_value_black,
      search800_root: p.search800.root_value_black,
    };
  };
  return {
    R_off: pos.filter((p) => p.played !== p.leader200).length,
    R_low: pos.filter((p) => p.search200.played_share < LOW_SHARE).length,
    B_diff: pos.filter((p) => p.leader800 !== p.leader200).length,
    T_top1: pos.filter((p) => p.played === p.teacher.top_move).length,
    T_prob_played: mean(pos.map((p) => p.teacher.prob[p.played])),
    S_prob_played: mean(pos.map((p) => p.student_raw.prior[p.played])),
    played_share_min: Math.min(...pos.map((p) => p.search200.played_share)),
    v19: vals(19),
    v9: vals(9),
    alt200: alt('leader200', 'search200'),
    alt800: alt('leader800', 'search800'),
  };
}

const SCALARS = {
  R_off: (m) => m.R_off,
  R_low: (m) => m.R_low,
  B_diff: (m) => m.B_diff,
  T_top1: (m) => m.T_top1,
  T_prob_played: (m) => m.T_prob_played,
  S_prob_played: (m) => m.S_prob_played,
  v19_student_raw: (m) => m.v19.student_raw,
  v19_teacher_raw: (m) => m.v19.teacher_raw,
  v19_search200_root: (m) => m.v19.search200_root,
  v19_search800_root: (m) => m.v19.search800_root,
  v9_student_raw: (m) => m.v9.student_raw,
  v9_teacher_raw: (m) => m.v9.teacher_raw,
  alt200_delta_student_raw: (m) => m.alt200.student_raw,
  alt200_delta_teacher_raw: (m) => m.alt200.teacher_raw,
  alt200_delta_search200: (m) => m.alt200.search,
  alt800_delta_student_raw: (m) => m.alt800.student_raw,
  alt800_delta_teacher_raw: (m) => m.alt800.teacher_raw,
  alt800_delta_search800: (m) => m.alt800.search,
};

function compare(Lm, Wm, Dm, opts) {
  const out = {};
  for (const [name, f] of Object.entries(SCALARS)) {
    const L = Lm.map(f).filter((x) => x !== null && Number.isFinite(x));
    const W = Wm.map(f).filter((x) => x !== null && Number.isFinite(x));
    const D = Dm.map(f);
    const diff = mean(L) !== null && mean(W) !== null ? mean(L) - mean(W) : null;
    out[name] = {
      mean_L: mean(L), n_L: L.length,
      mean_W: mean(W), n_W: W.length,
      D_values: D,
      diff_L_minus_W: diff,
      ci95: L.length && W.length ? bootstrap(L, W, (a, b) => mean(a) - mean(b), opts) : null,
    };
  }
  const anyRate = (f) => {
    const kL = Lm.filter(f).length;
    const kW = Wm.filter(f).length;
    return { L: [kL, Lm.length], W: [kW, Wm.length], D: [Dm.filter(f).length, Dm.length], wilson_L: wilson(kL, Lm.length), wilson_W: wilson(kW, Wm.length) };
  };
  out.any_R_off = anyRate((m) => m.R_off >= 1);
  out.any_R_low = anyRate((m) => m.R_low >= 1);
  out.any_B_diff = anyRate((m) => m.B_diff >= 1);
  const auc = {};
  for (const ply of ['v19', 'v9']) for (const who of ['student_raw', 'teacher_raw', 'search200_root', 'search800_root']) {
    const L = Lm.map((m) => m[ply][who]);
    const W = Wm.map((m) => m[ply][who]);
    auc[`${ply}_${who}`] = { auc: aucLowerIsLoss(L, W), ci95: bootstrap(Lm, Wm, (a, b) => aucLowerIsLoss(a.map((m) => m[ply][who]), b.map((m) => m[ply][who])), opts) };
  }
  // Paired over the same resamples: teacher minus student (raw), at ply 19.
  auc.v19_teacher_minus_student_raw = {
    diff: auc.v19_teacher_raw.auc - auc.v19_student_raw.auc,
    ci95: bootstrap(Lm, Wm, (a, b) => aucLowerIsLoss(a.map((m) => m.v19.teacher_raw), b.map((m) => m.v19.teacher_raw)) - aucLowerIsLoss(a.map((m) => m.v19.student_raw), b.map((m) => m.v19.student_raw)), opts),
  };
  out.auc = auc;
  return out;
}

/** The frozen signals (plan §6). They choose a follow-up; they do not assign causes to the losses. */
export function signals(cmp, nLosses) {
  const ratioOk = (s) => s.mean_L !== null && s.mean_W !== null && (s.mean_W === 0 ? s.mean_L > 0 : s.mean_L >= SIGNAL.RATIO * s.mean_W);
  const readout = cmp.any_R_low.L[0] >= SIGNAL.MIN_LOSSES && ratioOk(cmp.R_low) && cmp.R_low.ci95 && cmp.R_low.ci95.lo > 0;
  const budget = cmp.any_B_diff.L[0] >= SIGNAL.MIN_LOSSES && ratioOk(cmp.B_diff) && cmp.B_diff.ci95 && cmp.B_diff.ci95.lo > 0;
  const t = cmp.auc.v19_teacher_raw.auc;
  const s = cmp.auc.v19_student_raw.auc;
  const d = cmp.auc.v19_teacher_minus_student_raw.ci95;
  const teacher = t !== null && s !== null && t >= SIGNAL.AUC_TEACHER_MIN && s <= SIGNAL.AUC_STUDENT_MAX && d && d.lo > 0;
  const fired = [readout && 'readout', teacher && 'teacher', budget && 'budget'].filter(Boolean);
  const follow = {
    readout: 'a readout-setting change (e.g. a lower opening temperature) in a fresh match against current new hard',
    teacher: 'fresh training data from the student\'s own play, labelled by the teacher, fitted from the current student and tested against it on fresh games',
    budget: 'a search-budget change in a fresh match against current new hard',
  };
  return {
    readout: Boolean(readout), teacher: Boolean(teacher), budget: Boolean(budget), n_losses: nLosses,
    fired,
    recommended: fired.length ? fired[0] : null, // priority: readout, then teacher, then budget (plan §6)
    recommended_follow_up: fired.length ? follow[fired[0]] : 'no signal: the diagnostic does not single out a follow-up; the proposal\'s second step (a twixtbot benchmark card) stands as the default',
  };
}

export function analyse(students, teacher, groups, opts = {}) {
  const tKey = new Map(teacher.records.map((r) => [`${r.pair}/${r.ply}`, r]));
  const byGame = new Map();
  for (const s of students) {
    const t = tKey.get(`${s.pair}/${s.ply}`);
    if (!t) throw new Error(`no teacher record for pair ${s.pair} ply ${s.ply}`);
    for (const c of s.candidates) {
      if (!(c in t.after_value_black) || !(c in t.prob)) throw new Error(`teacher misses candidate ${c} at pair ${s.pair} ply ${s.ply}`);
    }
    const p = { ...s, teacher: t };
    if (!byGame.has(s.pair)) byGame.set(s.pair, []);
    byGame.get(s.pair).push(p);
  }
  const G = new Map(groups.games.map((g) => [g.pair, g]));
  const rows = [];
  for (const [pair, pos] of byGame) {
    if (pos.length !== PLIES.length) throw new Error(`pair ${pair}: ${pos.length} positions`);
    const g = G.get(pair);
    if (!g) throw new Error(`pair ${pair}: no group`);
    rows.push({ pair, outcome: g.outcome, race19: g.race19, ...gameMetrics(pos) });
  }
  if (rows.length !== groups.n_games) throw new Error(`${rows.length} games, groups have ${groups.n_games}`);
  const pick = (o, sub) => rows.filter((r) => r.outcome === o && (!sub || r.race19 === 'behind_2plus'));
  const all = compare(pick('L'), pick('W'), pick('D'), opts);
  const b2 = compare(pick('L', true), pick('W', true), pick('D', true), opts);
  // Position-level description by ply (not used by the signals).
  const byPly = {};
  for (const ply of PLIES) {
    const at = (o) => [...byGame.values()].flat().filter((p) => p.ply === ply && G.get(p.pair).outcome === o);
    const d = (o) => {
      const ps = at(o);
      return {
        n: ps.length,
        played_not_leader200: ps.filter((p) => p.played !== p.leader200).length,
        played_share_lt_10pct: ps.filter((p) => p.search200.played_share < LOW_SHARE).length,
        leader800_ne_leader200: ps.filter((p) => p.leader800 !== p.leader200).length,
        mean_student_raw_black: mean(ps.map((p) => p.student_raw.value_black)),
        mean_teacher_raw_black: mean(ps.map((p) => p.teacher.value_black)),
        mean_search200_root_black: mean(ps.map((p) => p.search200.root_value_black)),
        mean_search800_root_black: mean(ps.map((p) => p.search800.root_value_black)),
      };
    };
    byPly[ply] = { L: d('L'), W: d('W'), D: d('D') };
  }
  return {
    schema: 'htb-diag-o-analysis/1',
    n_games: rows.length,
    counts: { L: pick('L').length, W: pick('W').length, D: pick('D').length, B2_L: pick('L', true).length, B2_W: pick('W', true).length, B2_D: pick('D', true).length },
    bootstrap: { B: opts.B ?? BOOT_B, seed: opts.seed ?? BOOT_SEED, kind: 'stratified by outcome, games resampled, 95% percentile' },
    all_black_games: all,
    behind_2plus_at_19: b2,
    signals: signals(all, pick('L').length),
    by_ply: byPly,
    games: rows,
  };
}

function main() {
  const [cmd, dir] = process.argv.slice(2);
  if (cmd !== 'analyse' || !dir) {
    console.error('usage: node tests/htb_integration/diag_o_analyse.mjs analyse <run_dir>');
    process.exit(2);
  }
  const students = readFileSync(join(dir, 'student.jsonl'), 'utf8').trim().split('\n').map((l) => JSON.parse(l));
  const teacher = JSON.parse(readFileSync(join(dir, 'teacher.json'), 'utf8'));
  const groups = JSON.parse(readFileSync(join(ROOT, GROUPS_REL), 'utf8'));
  process.stdout.write(JSON.stringify(analyse(students, teacher, groups), null, 1) + '\n');
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) main();
