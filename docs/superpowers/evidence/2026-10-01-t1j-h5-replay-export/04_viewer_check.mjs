// H5 REPLAY EXPORT -- the viewer's OWN load / step / place / win path, under Node.
//
// For every file in 03_viewer_selection.json, mirrors assets/js/replay/replayController.js
// exactly: JSON.parse; the `moves` gate; resetGame()'s starting player; applyMove() =
// currentPlayer + forcePlacePeg per move; applyUpTo(i) for stepping; updateMeta()'s fields.
// Checks: the board holds exactly the recorded pegs; stepping to i shows i pegs; the
// viewer's OWN checkWin agrees with the file and fires no earlier than the last move; every
// displayed meta field is present and the labels are H5's. Rendering (WebGL) is NOT
// exercised here -- that is the live browser check.
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '../../../..');
const { default: TwixTGame } = await import(path.join(ROOT, 'assets/js/game/twixtGame.js'));
const files = JSON.parse(readFileSync(path.join(HERE, '03_viewer_selection.json'), 'utf8'));
const H5_NOTE = 'derived from the H5 results file; a viewer export, not a result';

function resetGame(replay) {
  const game = new TwixTGame();
  game.gameOver = true;
  const start = replay.starting_player ?? replay.meta?.starting_player ?? replay.moves[0].player;
  game.startingPlayer = start;
  game.currentPlayer = start;
  return game;
}

function applyUpTo(replay, index) {
  const game = resetGame(replay);
  let firstWinAt = null;
  for (let i = 0; i < index; i++) {
    game.currentPlayer = replay.moves[i].player;
    game.forcePlacePeg(replay.moves[i].row, replay.moves[i].col);
    if (game.winner !== null && firstWinAt === null) firstWinAt = i + 1;
  }
  return { game, firstWinAt };
}

let passed = 0;
for (const rel of files) {
  const replay = JSON.parse(readFileSync(path.join(ROOT, rel), 'utf8'));
  const problems = [];
  if (!replay.moves) problems.push('missing moves array');
  const n = replay.moves.length;
  const { game, firstWinAt } = applyUpTo(replay, n);
  const want = new Map(replay.moves.map((m) => [`${m.row},${m.col}`, m.player]));
  let held = 0;
  for (let r = 0; r < game.boardSize; r++) {
    for (let c = 0; c < game.boardSize; c++) {
      const v = game.board[r][c];
      if (v === null) continue;
      held++;
      if (want.get(`${r},${c}`) !== v) problems.push(`board ${r},${c} holds ${v}`);
    }
  }
  if (held !== n || game.pegs.length !== n) problems.push(`board holds ${held} pegs for ${n} moves`);
  for (const i of [0, 1, Math.floor(n / 2), n - 1, n]) {
    if (applyUpTo(replay, i).game.pegs.length !== i) problems.push(`stepping to ${i}`);
  }
  const fileWinner = replay.winner === 'draw' ? null : replay.winner;
  if (game.winner !== fileWinner) problems.push(`viewer winner ${game.winner} != file ${replay.winner}`);
  if (fileWinner !== null && firstWinAt !== n) problems.push(`viewer win first at move ${firstWinAt}, not ${n}`);
  const shown = {
    winner: replay.winner, reason: replay.meta?.reason, moves: replay.meta?.n_moves,
    seed: replay.seed, start: replay.starting_player, red: replay.meta?.players?.red,
    black: replay.meta?.players?.black, evidence: replay.meta?.evidence_note,
  };
  const missing = Object.entries(shown).filter(([, v]) => v === undefined || v === null || v === '');
  if (missing.length) problems.push(`meta not shown: ${missing.map(([k]) => k)}`);
  if (shown.evidence !== H5_NOTE) problems.push('evidence note is not H5\'s');
  if (problems.length === 0) passed++;
  console.log(`${problems.length ? 'FAIL' : 'PASS'}  ${path.basename(rel)}  arm ${replay.meta.arm}  ` +
              `${n} moves  reason ${shown.reason}  viewer winner ${game.winner ?? 'none'}  ` +
              `red="${shown.red}" black="${shown.black}"${problems.length ? '  ' + problems.join('; ') : ''}`);
}
console.log(`\nviewer path: ${passed}/${files.length} files pass`);
process.exit(passed === files.length ? 0 : 1);
