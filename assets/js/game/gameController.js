// assets/js/game/gameController.js
import TwixTGame from './twixtGame.js';
import Board3DRenderer from './board3DRenderer.js';
import TwixTAI from '../ai/search.js';
import { alphaZero } from '../ai/alphaZeroClient.js';
import { gameRecorder } from '../ui/gameRecorder.js';
import { winBar, WinBar } from '../ui/winBar.js';

export default class GameController {
  constructor() {
    this.game = new TwixTGame();
    this.renderer = null;

    // IMPORTANT: don't construct AI yet; wait until user picks 1-Player mode.
    this.ai = null;

    this.moveLog = [];
    this.useAlphaZero = false;

    // Two win bars: NN (raw single eval) and MCTS (search-based)
    this.winBarNN = winBar;                      // existing singleton, id='win-bar'
    this.winBarMCTS = new WinBar('win-bar-mcts'); // second bar, id='win-bar-mcts'

    this.init();

    // Check AlphaZero availability and wire win bars
    alphaZero.checkAvailability().then(available => {
      this.useAlphaZero = available;
      this.winBarNN.setEnabled(available);
      this.winBarMCTS.setEnabled(available);
      if (available) {
        // Wire live MCTS progress updates to the MCTS bar during AI search
        alphaZero.onProgress = (msg) => {
          if (msg.valueEstimate !== undefined && msg.toMove) {
            this.winBarMCTS.update(msg.valueEstimate, msg.toMove);
          }
        };
      }
    }).catch(() => {});
  }

  init() {
    const container = document.getElementById('canvas');
    this.renderer = new Board3DRenderer(container, this.game, this); // Pass reference to this controller

    // Add a small delay to ensure DOM is fully ready
    setTimeout(() => {
      this.setupEventListeners();
    }, 100);

    // Force initial button state
    const undoButton = document.getElementById('undo');
    if (undoButton) {
      undoButton.disabled = true;
      undoButton.setAttribute('disabled', 'disabled');
    }

    this.updateUI();
  }

  setupEventListeners() {
    console.log('Setting up event listeners...');

    // Mode selection buttons
    const twoP = document.getElementById('two-player-mode');
    console.log('2 Player button:', twoP);
    if (twoP) {
      twoP.addEventListener('click', () => {
        console.log('2 Player mode clicked!');
        this.startGame(false);
      });
    }

    const oneP = document.getElementById('one-player-mode');
    if (oneP) {
      oneP.addEventListener('click', () => {
        this.showDifficultySelection();
      });
    }

    // Difficulty selection buttons
    const easy = document.getElementById('easy-ai');
    if (easy) {
      easy.addEventListener('click', () => {
        this.startGame(true, 'easy');
      });
    }

    const med = document.getElementById('medium-ai');
    if (med) {
      med.addEventListener('click', () => {
        this.startGame(true, 'medium');
      });
    }

    const hard = document.getElementById('hard-ai');
    if (hard) {
      hard.addEventListener('click', () => {
        this.startGame(true, 'hard');
      });
    }

    // Control buttons
    const newGameBtn = document.getElementById('new-game');
    if (newGameBtn) {
      newGameBtn.addEventListener('click', () => {
        this.showModeSelection();
      });
    }

    const undoBtn = document.getElementById('undo');
    if (undoBtn) {
      undoBtn.addEventListener('click', () => {
        this.undo();
      });
    }

    const helpBtn = document.getElementById('help');
    if (helpBtn) {
      helpBtn.addEventListener('click', () => {
        const instructions = document.getElementById('instructions');
        if (instructions) instructions.classList.toggle('show');
      });
    }

    const playAgainBtn = document.getElementById('play-again');
    if (playAgainBtn) {
      playAgainBtn.addEventListener('click', () => {
        this.newGame();
        this.hideWinnerModal();
      });
    }

    // REC toggle
    const recBtn = document.getElementById('rec-toggle');
    const recDot = document.getElementById('rec-dot');
    const analysisEl = document.getElementById('analysis-status');
    const analysisElGame = document.getElementById('analysis-status-game');
    const recStatusBar = document.getElementById('rec-status-bar');

    if (recBtn) {
      if (gameRecorder.isEnabled) recDot.classList.add('active');

      recBtn.addEventListener('click', () => {
        const nowEnabled = gameRecorder.toggle();
        recDot.classList.toggle('active', nowEnabled);
      });

      gameRecorder.onAnalysisStatusChange = (status) => {
        const text = status === 'analyzing' ? 'Analyzing...'
          : status === 'ready' ? 'Ready'
          : '';
        const cls = `analysis-status ${status}`;
        if (analysisEl) { analysisEl.textContent = text; analysisEl.className = cls; }
        if (analysisElGame) { analysisElGame.textContent = text; analysisElGame.className = cls; }
        if (recStatusBar) recStatusBar.style.display = gameRecorder.state === 'RECORDING' ? 'inline' : 'none';
      };
    }

    // Show recording note in difficulty modal
    const recordingNote = document.getElementById('recording-note');

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
      if (e.key === 'u' || e.key === 'U') {
        this.undo();
      } else if (e.key === 'n' || e.key === 'N') {
        this.newGame();
      } else if (e.key === 'h' || e.key === 'H') {
        const instructions = document.getElementById('instructions');
        if (instructions) instructions.classList.toggle('show');
      }
    });
  }

  updateUI() {
    const playerIndicator = document.getElementById('current-player');
    const playerName = document.getElementById('player-name');
    const moveCount = document.getElementById('move-count');

    if (this.game.currentPlayer === 'red') {
      if (playerIndicator)
        playerIndicator.className = 'current-player player-red';
      if (playerName)
        playerName.textContent = 'Red (Cross Goal Lines: Top ↔ Bottom)';
    } else {
      if (playerIndicator)
        playerIndicator.className = 'current-player player-black';
      if (playerName)
        playerName.textContent = 'Black (Cross Goal Lines: Left ↔ Right)';
    }

    if (moveCount) moveCount.textContent = `Move ${this.game.moveCount}`;

    const undoButton = document.getElementById('undo');
    const shouldDisable = this.game.moveHistory.length === 0;

    if (undoButton) {
      if (shouldDisable) {
        undoButton.disabled = true;
        undoButton.setAttribute('disabled', 'disabled');
        undoButton.style.opacity = '0.3';
        undoButton.style.cursor = 'not-allowed';
        undoButton.style.pointerEvents = 'none';
      } else {
        undoButton.disabled = false;
        undoButton.removeAttribute('disabled');
        undoButton.style.opacity = '1';
        undoButton.style.cursor = 'pointer';
        undoButton.style.pointerEvents = 'auto';
      }

      // Force a repaint (keeps your original behavior)
      void undoButton.offsetHeight;
    }
  }

  showWinner() {
    const modal = document.getElementById('winner-modal');
    const winnerText = document.getElementById('winner-text');

    const winnerName = this.game.winner === 'red' ? 'Red' : 'Black';

    // Show alert as immediate feedback
    alert(`🎉 ${winnerName} Player Wins! 🎉`);

    if (!modal || !winnerText) {
      console.error('Winner modal elements not found');
      return;
    }

    winnerText.textContent = `${winnerName} Wins!`;
    winnerText.style.color = this.game.winner === 'red' ? '#ff4757' : '#3498db';

    modal.style.display = 'block';
    modal.classList.add('show');
  }

  hideWinnerModal() {
    const modal = document.getElementById('winner-modal');
    if (modal) {
      modal.classList.remove('show');
      modal.style.display = 'none';
    }
  }

  showModeSelection() {
    const mode = document.getElementById('game-mode-modal');
    const diff = document.getElementById('difficulty-modal');
    if (mode) mode.style.display = 'flex';
    if (diff) diff.style.display = 'none';
    this.hideWinnerModal();
  }

  showDifficultySelection() {
    const mode = document.getElementById('game-mode-modal');
    const diff = document.getElementById('difficulty-modal');
    if (mode) mode.style.display = 'none';
    if (diff) diff.style.display = 'flex';
    const recordingNote = document.getElementById('recording-note');
    if (recordingNote) recordingNote.style.display = gameRecorder.isEnabled ? 'block' : 'none';
  }

  startGame(isAI, difficulty = 'medium') {
    // Hide modals
    const mode = document.getElementById('game-mode-modal');
    const diff = document.getElementById('difficulty-modal');
    if (mode) mode.style.display = 'none';
    if (diff) diff.style.display = 'none';

    // Reset and configure game
    this.game.reset();
    this.game.setGameMode(isAI, difficulty);

    // Only construct AI if we actually need it
    this.ai = isAI ? new TwixTAI(this.game) : null;

    // Randomize starting player
    const startsBlack = Math.random() < 0.5;
    if (startsBlack) {
      this.game.startingPlayer = 'black';
      this.game.currentPlayer = 'black';
    } else {
      this.game.startingPlayer = 'red';
      this.game.currentPlayer = 'red';
    }

    this.moveLog = [];
    this.renderer.updateBoard();
    this.updateUI();

    // Update UI to show game mode
    const playerName = document.getElementById('player-name');
    if (playerName) {
      if (isAI) {
        const diffLabel =
          difficulty.charAt(0).toUpperCase() + difficulty.slice(1);
        const starter = startsBlack ? `Black (${diffLabel} AI)` : 'Red (Human)';
        playerName.textContent = `Red (Human) vs Black (${diffLabel} AI) – ${starter} starts`;
      } else {
        playerName.textContent = startsBlack
          ? 'Black moves first (Human vs Human)'
          : 'Red moves first (Human vs Human)';
      }
    }

    // Reset both win bars
    this.winBarNN.clear();
    this.winBarMCTS.clear();
    this.winBarNN.setEnabled(this.useAlphaZero);
    this.winBarMCTS.setEnabled(this.useAlphaZero);

    // Start recording if enabled and 1-Player mode
    if (isAI && gameRecorder.isEnabled) {
      gameRecorder.startRecording(this.game, difficulty, alphaZero);
      const recBar = document.getElementById('rec-status-bar');
      if (recBar) recBar.style.display = 'inline';
    }

    // If AI should open, trigger immediately
    if (isAI && startsBlack && this.game.aiPlayer === 'black') {
      setTimeout(() => this.makeAIMove(), 300);
    } else if (isAI && gameRecorder.state === 'RECORDING') {
      // Human goes first -- start precompute
      gameRecorder.markTurnStart();
      gameRecorder.precomputeAnalysis(this.game, alphaZero);
    }
  }

  newGame() {
    this.showModeSelection();
  }

  // Handle player move and trigger AI if needed
  async onPlayerMove(success) {
    if (!success) return;

    // Update NN bar after human move (quick single eval)
    if (this.useAlphaZero && !this.game.gameOver) {
      alphaZero.evaluate(this.game).then(valueRed => {
        if (valueRed !== null) this.winBarNN.updateRed(valueRed);
      }).catch(() => {});
    }
    // Update MCTS bar from precompute if recording captured one
    if (gameRecorder.state === 'RECORDING' && gameRecorder._lastPrecomputeHash) {
      const cached = gameRecorder.analysisCache.get(gameRecorder._lastPrecomputeHash);
      if (cached?.status === 'completed' && cached?.root?.root_value !== undefined) {
        // root_value is from side-to-move perspective; convert to Red
        const toMove = this.game.currentPlayer === this.game.aiPlayer
          ? (this.game.aiPlayer === 'red' ? 'black' : 'red')  // human just moved, it's AI's turn now
          : this.game.currentPlayer;
        // Actually: the precompute ran BEFORE the human moved, so root_value
        // is from the human's perspective (they were to_move)
        const humanColor = this.game.aiPlayer === 'black' ? 'red' : 'black';
        const valueRed = humanColor === 'red' ? cached.root.root_value : -cached.root.root_value;
        this.winBarMCTS.updateRed(valueRed);
      }
    }
    if (this.game.gameOver && this.game.winner) {
      const defValue = this.game.winner === 'red' ? 1 : -1;
      this.winBarNN.updateRed(defValue);
      this.winBarMCTS.updateRed(defValue);
    }

    // Finalize recording if human just won
    if (this.game.gameOver && this.game.winner) {
      if (gameRecorder.state === 'RECORDING') {
        const saveResult = await gameRecorder.finalize(this.game, alphaZero);
        if (saveResult?.ok) console.log('Game saved:', saveResult.path);
        else console.warn('Game save failed:', saveResult?.error);
      }
    }

    if (
      this.game.isAIGame &&
      this.game.currentPlayer === this.game.aiPlayer &&
      !this.game.gameOver
    ) {
      // Delay AI move slightly for better UX
      setTimeout(() => {
        this.makeAIMove();
      }, 500);
    }
  }

  // Make AI move
  async makeAIMove() {
    if (
      this.game.gameOver ||
      !this.game.isAIGame ||
      this.game.currentPlayer !== this.game.aiPlayer
    ) {
      return;
    }

    if (!this.ai) return; // guard

    const move = this.ai.getBestMove();
    if (move) {
      const player = this.game.currentPlayer;
      const success = this.game.placePeg(move.row, move.col);
      if (success) {
        this.recordMove('ai', player, move.row, move.col);
        this.renderer.updateBoard();
        this.updateUI();

        // Check for AI win
        if (this.game.gameOver && this.game.winner) {
          const defValue = this.game.winner === 'red' ? 1 : -1;
          this.winBarNN.updateRed(defValue);
          this.winBarMCTS.updateRed(defValue);
          if (gameRecorder.state === 'RECORDING') {
            const saveResult = await gameRecorder.finalize(this.game, alphaZero);
            if (saveResult?.ok) console.log('Game saved:', saveResult.path);
            else console.warn('Game save failed:', saveResult?.error);
          }
          setTimeout(() => this.showWinner(), 200);
        }

        // After AI move, trigger precompute for human's next turn
        if (gameRecorder.state === 'RECORDING' && !this.game.gameOver) {
          gameRecorder.markTurnStart();
          gameRecorder.precomputeAnalysis(this.game, alphaZero);
        }
      }
    }
  }

  undo() {
    if (gameRecorder.state === 'RECORDING') {
      gameRecorder.truncateToMove(this.game.moveCount);
    }
    if (this.game.undo()) {
      this.renderer.updateBoard();

      // Direct UI update - same as in 3D renderer
      const moveCount = document.getElementById('move-count');
      if (moveCount) {
        moveCount.textContent = `Move ${this.game.moveCount}`;
      }

      const playerIndicator = document.getElementById('current-player');
      const playerName = document.getElementById('player-name');
      if (playerIndicator && playerName) {
        if (this.game.currentPlayer === 'red') {
          playerIndicator.className = 'current-player player-red';
          playerName.textContent = 'Red (Cross Goal Lines: Top ↔ Bottom)';
        } else {
          playerIndicator.className = 'current-player player-black';
          playerName.textContent = 'Black (Cross Goal Lines: Left ↔ Right)';
        }
      }

      const undoButton = document.getElementById('undo');
      if (undoButton) {
        if (this.game.moveHistory.length === 0) {
          undoButton.disabled = true;
          undoButton.setAttribute('disabled', 'disabled');
          undoButton.style.opacity = '0.3';
          undoButton.style.cursor = 'not-allowed';
          undoButton.style.pointerEvents = 'none';
        } else {
          undoButton.disabled = false;
          undoButton.removeAttribute('disabled');
          undoButton.style.opacity = '1';
          undoButton.style.cursor = 'pointer';
          undoButton.style.pointerEvents = 'auto';
        }
      }
    }
  }

  recordMove(source, player, row, col) {
    if (!this.moveLog) this.moveLog = [];

    // Record human moves in GameRecorder
    if (source === 'human' && gameRecorder.state === 'RECORDING') {
      gameRecorder.recordMove(this.game, 'human', player, row, col);
    }

    const entry = {
      move: this.moveLog.length + 1,
      source,
      player,
      row,
      col,
    };
    this.moveLog.push(entry);

    console.log(
      `[Move ${entry.move}] ${player.toUpperCase()} (${source}) -> (${row}, ${col})`
    );
  }
}
