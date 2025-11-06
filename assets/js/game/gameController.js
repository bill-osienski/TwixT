// assets/js/game/gameController.js
import TwixTGame from './twixtGame.js';
import Board3DRenderer from './board3DRenderer.js';
import TwixTAI from '../ai/search.js';

export default class GameController {
  constructor() {
    this.game = new TwixTGame();
    this.renderer = null;

    // IMPORTANT: don't construct AI yet; wait until user picks 1-Player mode.
    this.ai = null;

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
      // eslint-disable-next-line no-unused-expressions
      undoButton.offsetHeight;
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

    // If AI should open, trigger immediately
    if (isAI && startsBlack && this.game.aiPlayer === 'black') {
      setTimeout(() => this.makeAIMove(), 300);
    }
  }

  newGame() {
    this.showModeSelection();
  }

  // Handle player move and trigger AI if needed
  onPlayerMove(success) {
    if (
      success &&
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
  makeAIMove() {
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
          setTimeout(() => this.showWinner(), 200);
        }
      }
    }
  }

  undo() {
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
