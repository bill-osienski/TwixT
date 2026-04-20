import TwixTGame from '../game/twixtGame.js';
import Board3DRenderer from '../game/board3DRenderer.js';

const fileInput = document.getElementById('file-input');
const btnPrev = document.getElementById('btn-prev');
const btnPlay = document.getElementById('btn-play');
const btnNext = document.getElementById('btn-next');
const slider = document.getElementById('turn-slider');
const speedSelect = document.getElementById('speed');
const statusText = document.getElementById('status-text');

const metaWinner = document.getElementById('meta-winner');
const metaReason = document.getElementById('meta-reason');
const metaMoves = document.getElementById('meta-moves');
const metaSeed = document.getElementById('meta-seed');
const metaDepth = document.getElementById('meta-depth');
const metaStart = document.getElementById('meta-start');

const container = document.getElementById('canvas');

let renderer = null;
let game = null;
let replay = null;
let currentIndex = 0;
let playing = false;
let timer = null;

function initRenderer() {
  game = new TwixTGame();
  game.gameOver = true;
  renderer = new Board3DRenderer(container, game, null);
}

function resetGame() {
  game = new TwixTGame();
  game.gameOver = true;
  if (replay && replay.moves && replay.moves.length > 0) {
    // Use starting_player from file if available, otherwise infer from first move
    const startPlayer = replay.starting_player ?? replay.meta?.starting_player ?? replay.moves[0].player;
    game.startingPlayer = startPlayer;
    game.currentPlayer = startPlayer;
  }
  renderer.game = game;
}

function applyMove(move) {
  game.currentPlayer = move.player;
  // Use forcePlacePeg to show exactly what was recorded (no validation)
  // This helps debug issues like color swap bugs or illegal moves in training
  game.forcePlacePeg(move.row, move.col);
}

function applyUpTo(index) {
  resetGame();
  for (let i = 0; i < index; i++) {
    applyMove(replay.moves[i]);
  }
  renderer.updateBoard();
  updateStatus();
}

function updateStatus() {
  if (!replay) {
    statusText.textContent = 'No replay loaded.';
    return;
  }
  statusText.textContent = `Turn ${currentIndex} / ${replay.moves.length}`;
  slider.value = String(currentIndex);
}

function updateMeta() {
  metaWinner.textContent = replay?.winner || '-';
  metaReason.textContent = replay?.meta?.reason || replay?.reason || '-';
  metaMoves.textContent = replay?.meta?.n_moves ?? replay?.total_moves ?? replay?.moves?.length ?? '-';
  metaSeed.textContent = replay?.seed ?? '-';
  metaDepth.textContent = replay?.depth ?? replay?.meta?.simulations ?? '-';
  metaStart.textContent = replay?.starting_player ?? replay?.meta?.starting_player ?? '-';
}

function setReplay(data) {
  replay = data;
  currentIndex = 0;
  slider.max = String(replay.moves.length);
  updateMeta();
  applyUpTo(0);
}

function step(delta) {
  if (!replay) return;
  currentIndex = Math.max(0, Math.min(replay.moves.length, currentIndex + delta));
  applyUpTo(currentIndex);
}

function play() {
  if (!replay) return;
  playing = true;
  btnPlay.textContent = 'Pause';
  const interval = Number(speedSelect.value);
  timer = setInterval(() => {
    if (currentIndex >= replay.moves.length) {
      pause();
      return;
    }
    step(1);
  }, interval);
}

function pause() {
  playing = false;
  btnPlay.textContent = 'Play';
  if (timer) {
    clearInterval(timer);
    timer = null;
  }
}

fileInput.addEventListener('change', async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  const text = await file.text();
  const data = JSON.parse(text);
  if (!data.moves) {
    alert('Invalid replay file: missing moves array');
    return;
  }
  setReplay(data);
});

btnPrev.addEventListener('click', () => step(-1));
btnNext.addEventListener('click', () => step(1));
btnPlay.addEventListener('click', () => {
  if (playing) pause();
  else play();
});

slider.addEventListener('input', (event) => {
  if (!replay) return;
  currentIndex = Number(event.target.value);
  applyUpTo(currentIndex);
});

speedSelect.addEventListener('change', () => {
  if (playing) {
    pause();
    play();
  }
});

initRenderer();
