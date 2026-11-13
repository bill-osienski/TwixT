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



// Frozen input for Diagnostic O (written before any model load): per Black game, the outcome and the race group after ply 19.
const files = readdirSync(dir).filter((f) => f.endsWith('_game_1.json')).sort();
const games = [];
for (const f of files) {
  const g = JSON.parse(readFileSync(join(dir, f), 'utf8'));
  if (g.new_colour !== 'black' || g.stage !== 'match') throw new Error(`${f}: not a match game with the student as Black`);
  let s = new TwixtState();
  for (let i = 0; i <= 19; i++) s = s.applyMove(g.moves[i]);
  const dn = dist(s, 'black'), dl = dist(s, 'red');
  const diff = dn - dl;
  const race19 = diff >= 2 ? 'behind_2plus' : diff === 1 ? 'behind_1' : diff === 0 ? 'level' : 'ahead';
  const outcome = g.result === 'draw' ? 'D' : g.result === 'black' ? 'W' : 'L';
  games.push({ file: f, pair: g.pair_index, outcome, plies: g.ply_count, race19, dist_black: dn, dist_red: dl });
}
const n = (o) => games.filter((g) => g.outcome === o).length;
process.stdout.write(JSON.stringify({ schema: 'htb-diag-o-groups/1', n_games: games.length, W: n('W'), L: n('L'), D: n('D'), games }, null, 1) + '\n');
