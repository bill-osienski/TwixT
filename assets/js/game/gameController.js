// assets/js/game/gameController.js
import TwixTGame from './twixtGame.js';
import Board3DRenderer from './board3DRenderer.js';
import TwixTAI from '../ai/search.js';
import { alphaZero } from '../ai/alphaZeroClient.js';
import { winBar } from '../ui/winBar.js';
import { gameRecorder } from '../ui/gameRecorder.js';

export default class GameController {
  constructor() {
    this.game = new TwixTGame();
    this.renderer = null;

    // IMPORTANT: don't construct AI yet; wait until user picks 1-Player mode.
    this.ai = null;

    // AlphaZero integration
    this.alphaZero = alphaZero;
    this.winBar = winBar;
    this.useAlphaZero = false; // Will be set based on server availability

    this.moveLog = [];
    this.init();
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

    // Check AlphaZero availability
    this._checkAlphaZero();

    this.updateUI();
  }

  async _checkAlphaZero() {
    try {
      const available = await this.alphaZero.checkAvailability();
      this.useAlphaZero = available;
      if (available) {
        console.log('AlphaZero server detected - using neural network AI');
        // Enable win bar when AlphaZero is available
        this.winBar.setEnabled(true);
      } else {
        console.log('AlphaZero server not available - using heuristic AI');
        this.winBar.setEnabled(false);
      }
    } catch (err) {
      console.log('AlphaZero check failed:', err.message);
      this.useAlphaZero = false;
    }
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
  }

  async startGame(isAI, difficulty = 'medium') {
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

    // Reset win bar for new game
    if (this.winBar) {
      this.winBar.clear();
      // Enable win bar only when AlphaZero is available
      this.winBar.setEnabled(this.useAlphaZero);
    }

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

    // Start recording if enabled and 1-Player mode
    if (isAI && gameRecorder.isEnabled) {
      await gameRecorder.startRecording(this.game, difficulty, this.alphaZero);
    }

    // If human goes first, trigger precompute for the first move
    if (isAI && gameRecorder.state === 'RECORDING' && this.game.currentPlayer !== this.game.aiPlayer) {
      gameRecorder.markTurnStart();
      gameRecorder.precomputeAnalysis(this.game, this.alphaZero);
    }

    // If AI should open, trigger immediately
    if (isAI && startsBlack && this.game.aiPlayer === 'black') {
      setTimeout(() => this.makeAIMove(), 300);
    }
  }

  newGame() {
    this.showModeSelection();
  }

  // Handle player move and trigger AI if needed
  async onPlayerMove(success) {
    if (success) {
      // Update win bar after player move
      this._updateWinBar();

      // Finalize recording if human just won
      if (this.game.gameOver && this.game.winner) {
        if (gameRecorder.state === 'RECORDING') {
          const saveResult = await gameRecorder.finalize(this.game, this.alphaZero);
          if (saveResult?.ok) {
            console.log('Game saved:', saveResult.path);
          } else {
            console.warn('Game save failed:', saveResult?.error);
          }
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
  }

  // Update win bar with current position evaluation
  async _updateWinBar() {
    if (!this.winBar.isEnabled() || !this.useAlphaZero) return;
    if (this.game.gameOver) {
      // Show final result (already in Red's perspective)
      const valueRed = this.game.winner === 'red' ? 1 : -1;
      this.winBar.updateRed(valueRed);
      return;
    }

    try {
      const valueRed = await this.alphaZero.evaluate(this.game);
      if (valueRed !== null) {
        this.winBar.updateRed(valueRed);
      }
    } catch (err) {
      // Silently ignore evaluation errors
    }
  }

  // Make AI move - uses AlphaZero when available, falls back to heuristics
  async makeAIMove() {
    if (
      this.game.gameOver ||
      !this.game.isAIGame ||
      this.game.currentPlayer !== this.game.aiPlayer
    ) {
      return;
    }

    // Cancel any previous in-flight request
    this.alphaZero.cancelLast();

    const isRecording = gameRecorder.state === 'RECORDING';
    let move, source, result;

    // Try AlphaZero first if available
    if (this.useAlphaZero) {
      try {
        result = await this.alphaZero.getMove(
          this.game,
          this.game.aiDifficulty,
          { includeVisits: isRecording }
        );
        move = result.move;
        source = result.source;

        // Update win bar with Red-perspective value from server
        if (
          result.valueRed !== null &&
          result.valueRed !== undefined &&
          this.winBar.isEnabled()
        ) {
          this.winBar.updateRed(result.valueRed);
        }
      } catch (err) {
        console.warn('AlphaZero failed, falling back to heuristics:', err);
        this.useAlphaZero = false;
      }
    }

    // Fall back to heuristic AI
    if (!move) {
      if (!this.ai) return; // guard
      move = this.ai.getBestMove();
      source = 'heuristics';
    }

    if (move) {
      const player = this.game.currentPlayer;
      const success = this.game.placePeg(move.row, move.col);
      if (success) {
        this.recordMove(source, player, move.row, move.col);

        // Record AI move in GameRecorder
        if (isRecording && result?.visits) {
          const aiPacket = gameRecorder.buildAIMovePacket(this.game, result);
          gameRecorder.recordMove(this.game, 'alphazero', player, move.row, move.col, aiPacket);
        }

        this.renderer.updateBoard();
        this.updateUI();

        // Check for AI win
        if (this.game.gameOver && this.game.winner) {
          // Finalize recording before showing winner
          if (gameRecorder.state === 'RECORDING') {
            const saveResult = await gameRecorder.finalize(this.game, this.alphaZero);
            if (saveResult?.ok) {
              console.log('Game saved:', saveResult.path);
            } else {
              console.warn('Game save failed:', saveResult?.error);
            }
          }
          setTimeout(() => this.showWinner(), 200);
        }

        // After AI move, if it's human's turn, trigger precompute
        if (gameRecorder.state === 'RECORDING' && !this.game.gameOver) {
          gameRecorder.markTurnStart();
          gameRecorder.precomputeAnalysis(this.game, this.alphaZero);
        }
      }
    }
  }

  undo() {
    // Cancel any in-flight AI request (user is undoing during AI thinking)
    this.alphaZero.cancelLast();

    if (this.game.undo()) {
      // Truncate recorded moves to match the new game state
      if (gameRecorder.state === 'RECORDING') {
        gameRecorder.truncateToMove(this.game.moveCount);
      }

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

    // Record human moves in GameRecorder (AI moves are recorded in makeAIMove)
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
