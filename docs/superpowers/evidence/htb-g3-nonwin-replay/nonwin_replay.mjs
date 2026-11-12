// Rules-only replay of the G3 match games (no model, no search). Read-only.
// Usage: node g3_nonwin_replay.mjs <repo_root>
import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';

const root = process.argv[2];
const { TwixtState } = await import(join(root, 'server/gameLogic.js'));
const dir = join(root, 'docs/superpowers/evidence/htb-g3-match/games');
const N = 24;
const KN = [[1,2],[2,1],[-1,2],[-2,1],[1,-2],[2,-1],[-1,-2],[-2,-1]];
const INF = 1e9;

// Heuristic: fewest extra pegs `player` needs to join its goal edges, ignoring the opponent's replies.
// Nodes are holes; own peg costs 0, a hole `player` may legally take costs 1, opponent pegs are walls.
// A knight step is usable if it is an existing own bridge, or (when at least one end is empty) it crosses no
// existing bridge. Two own pegs a knight apart without a bridge were blocked when placed, so that step is unusable.
function dist(s, player) {
  const own = (r, c) => s.pegs.get(`${r},${c}`) === player;
  const opp = (r, c) => { const p = s.pegs.get(`${r},${c}`); return p !== undefined && p !== player; };
  const placeable = (r, c) => {
    if (r < 0 || r >= N || c < 0 || c >= N) return false;
    if ((r === 0 || r === N - 1) && (c === 0 || c === N - 1)) return false;
    if (player === 'red' && (c === 0 || c === N - 1)) return false;
    if (player === 'black' && (r === 0 || r === N - 1)) return false;
    return true;
  };
  const cost = (r, c) => (own(r, c) ? 0 : opp(r, c) || !placeable(r, c) ? INF : 1);
  const isStart = (r, c) => (player === 'red' ? r === 0 : c === 0);
  const isGoal = (r, c) => (player === 'red' ? r === N - 1 : c === N - 1);
  const d = new Array(N * N).fill(INF);
  const dq = [];
  for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
    if (!isStart(r, c)) continue;
    const k = cost(r, c); if (k >= INF) continue;
    d[r * N + c] = k; dq.push([r, c]);
  }
  // Dijkstra with small integer weights (bucketed).
  const buckets = [[]]; for (const [r, c] of dq) { (buckets[d[r*N+c]] ||= []).push([r, c]); }
  for (let b = 0; b < buckets.length; b++) {
    const q = buckets[b] || [];
    for (let i = 0; i < q.length; i++) {
      const [r, c] = q[i];
      if (d[r * N + c] !== b) continue;
      if (isGoal(r, c)) return b;
      for (const [dr, dc] of KN) {
        const r2 = r + dr, c2 = c + dc;
        if (r2 < 0 || r2 >= N || c2 < 0 || c2 >= N) continue;
        const k = cost(r2, c2); if (k >= INF) continue;
        const bothOwn = own(r, c) && own(r2, c2);
        let ok;
        if (bothOwn) ok = s.bridges.has(`${r},${c}-${r2},${c2}`) || s.bridges.has(`${r2},${c2}-${r},${c}`);
        else ok = !s._crossesExistingBridge(r, c, r2, c2);
        if (!ok) continue;
        const nd = b + k;
        if (nd < d[r2 * N + c2]) { d[r2 * N + c2] = nd; (buckets[nd] ||= []).push([r2, c2]); }
      }
    }
  }
  return INF;
}

const files = readdirSync(dir).filter((f) => f.endsWith('.json')).sort();
const games = files.map((f) => JSON.parse(readFileSync(join(dir, f), 'utf8')));
const fmt = (x) => (x >= INF ? '∞' : String(x));
const out = [];
for (const g of games) {
  const nc = g.new_colour, lc = nc === 'red' ? 'black' : 'red';
  let s = new TwixtState();
  const traj = [];
  for (const [i, m] of g.moves.entries()) {
    s = s.applyMove(m);
    traj.push([dist(s, nc), dist(s, lc)]);
  }
  const res = s.gameResult();
  if (res !== g.result) throw new Error(`result mismatch ${g.pair_index}/${g.game_in_pair}: ${res} vs ${g.result}`);
  // Last ply at which the new side was ahead-or-level in the race, adjusting for who moves next.
  // After ply p (0-based), the side to move next is ply p+1's mover. Being "behind" = own dist > opp dist,
  // or equal while the opponent is to move.
  let lastOk = -1, firstInfNew = null, firstInfOpp = null;
  for (let p = 0; p < traj.length; p++) {
    const [dn, dl] = traj[p];
    const nextMover = (p + 1) % 2 === 0 ? 'red' : 'black';
    const behind = dn > dl || (dn === dl && nextMover === lc);
    if (!behind) lastOk = p;
    if (dn >= INF && firstInfNew === null) firstInfNew = p;
    if (dl >= INF && firstInfOpp === null) firstInfOpp = p;
  }
  out.push({ pair: g.pair_index, gip: g.game_in_pair, nc, result: g.result, term: g.termination, plies: g.ply_count,
    score: g.new_score, moves: g.moves, traj, lastOk, firstInfNew, firstInfOpp });
}

const nonwins = out.filter((o) => (o.term === 'win' && o.result !== o.nc) || o.result === 'draw');
const losses = out.filter((o) => o.term === 'win' && o.result !== o.nc);
const draws = out.filter((o) => o.result === 'draw');
const wins = out.filter((o) => o.result === o.nc);
console.log(`games ${out.length}; wins ${wins.length}; losses ${losses.length} (as red ${losses.filter(o=>o.nc==='red').length}, as black ${losses.filter(o=>o.nc==='black').length}); draws ${draws.length} (as red ${draws.filter(o=>o.nc==='red').length})`);
console.log('replayed results all equal recorded results');

console.log('\n== LOSSES: race trajectory (new dist / legacy dist after each ply, heuristic) ==');
for (const o of losses) {
  const turn = o.lastOk + 1; // first ply after which the new side stayed behind to the end
  const t = o.traj.map(([a, b], p) => `${p}:${fmt(a)}/${fmt(b)}`);
  console.log(`pair ${o.pair} g${o.gip} new=${o.nc} plies ${o.plies}; behind for good from ply ${turn} (${turn < 20 ? 'opening phase, plies 0-19' : 'halving phase, ply 20+'})`);
  console.log('   ' + t.slice(Math.max(0, turn - 6), turn + 8).join(' '));
}
console.log('\n== WINS: ply from which the LEGACY side stayed behind (for comparison) ==');
{
  const turns = [];
  for (const o of wins) {
    let lastOkL = -1;
    for (let p = 0; p < o.traj.length; p++) {
      const [dn, dl] = o.traj[p]; const nextMover = (p + 1) % 2 === 0 ? 'red' : 'black';
      const lc = o.nc === 'red' ? 'black' : 'red';
      const behindL = dl > dn || (dl === dn && nextMover === o.nc);
      if (!behindL) lastOkL = p;
    }
    turns.push({ nc: o.nc, t: lastOkL + 1 });
  }
  for (const col of ['red', 'black']) {
    const ts = turns.filter((x) => x.nc === col).map((x) => x.t).sort((a, b) => a - b);
    const q = (f) => ts[Math.floor(f * (ts.length - 1))];
    console.log(`new=${col}: n ${ts.length}, decided-at ply median ${q(0.5)}, p10 ${q(0.1)}, p90 ${q(0.9)}, before ply 20: ${ts.filter((t)=>t<20).length}`);
  }
}

console.log('\n== DRAWS ==');
for (const o of draws) {
  console.log(`pair ${o.pair} g${o.gip} new=${o.nc} plies ${o.plies} term ${o.term}; new side unconnectable from ply ${o.firstInfNew}, legacy from ply ${o.firstInfOpp}`);
}

console.log('\n== OPENINGS when new plays BLACK (legacy red, deterministic) ==');
const asBlack = out.filter((o) => o.nc === 'black');
const r0 = new Map(); for (const o of asBlack) r0.set(JSON.stringify(o.moves[0]), (r0.get(JSON.stringify(o.moves[0])) || 0) + 1);
console.log('legacy first move(s):', [...r0]);
const byReply = new Map();
for (const o of asBlack) { const k = JSON.stringify(o.moves[1]); const e = byReply.get(k) || { n: 0, w: 0, l: 0, d: 0 }; e.n++; if (o.result === 'draw') e.d++; else if (o.result === o.nc) e.w++; else e.l++; byReply.set(k, e); }
console.log('new reply at ply 1 -> n / wins / losses / draws:');
for (const [k, e] of [...byReply].sort((a, b) => b[1].n - a[1].n)) console.log(`  ${k}: ${e.n} / ${e.w} / ${e.l} / ${e.d}`);

console.log('\n== Shared prefixes among losses as black ==');
const lb = losses.filter((o) => o.nc === 'black');
for (let i = 0; i < lb.length; i++) {
  let best = 0, with_ = null;
  for (let j = 0; j < lb.length; j++) { if (i === j) continue; let k = 0; while (k < Math.min(lb[i].moves.length, lb[j].moves.length) && lb[i].moves[k][0] === lb[j].moves[k][0] && lb[i].moves[k][1] === lb[j].moves[k][1]) k++; if (k > best) { best = k; with_ = lb[j].pair; } }
  // also against all black games that the new side won
  let bestW = 0;
  for (const o of asBlack) { if (o === lb[i] || o.result !== o.nc) continue; let k = 0; while (k < Math.min(lb[i].moves.length, o.moves.length) && lb[i].moves[k][0] === o.moves[k][0] && lb[i].moves[k][1] === o.moves[k][1]) k++; if (k > bestW) bestW = k; }
  console.log(`  loss pair ${lb[i].pair}: longest shared prefix with another loss ${best} plies (pair ${with_}); with a black WIN ${bestW} plies`);
}
console.log('\n== Losses: move lists ==');
for (const o of nonwins) console.log(`pair ${o.pair} g${o.gip} new=${o.nc} ${o.result}/${o.term} ${o.plies}: ${o.moves.slice(0, 60).map(([r, c]) => `${r},${c}`).join(' ')}${o.plies > 60 ? ' …' : ''}`);
