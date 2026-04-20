/**
 * Win prediction bar component.
 *
 * Displays neural network's evaluation as a visual bar:
 * - Red side = NN thinks red is winning
 * - Black side = NN thinks black is winning
 * - Center = even position
 *
 * The bar shows the position from Red's perspective (positive = red winning).
 *
 * Usage:
 *   import { WinBar, winBar } from './winBar.js';
 *
 *   // Update with new value
 *   winBar.update(0.3, 'red');  // Red has +0.3 advantage, it's red's turn
 *
 *   // Enable/disable
 *   winBar.setEnabled(true);
 */

export class WinBar {
  constructor(containerId = 'win-bar') {
    this.containerId = containerId;
    this.container = null;
    this.value = 0; // -1 to +1, from red's perspective
    this.enabled = false;
    this.toMove = 'red';

    // Elements (created in _createElements)
    this.redFill = null;
    this.blackFill = null;
    this.percentageEl = null;

    // Defer initialization until DOM is ready
    if (typeof document !== 'undefined') {
      if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => this._init());
      } else {
        this._init();
      }
    }
  }

  _init() {
    this.container = document.getElementById(this.containerId);
    if (this.container) {
      this._createElements();
      this._injectStyles();
    }
  }

  _createElements() {
    this.container.innerHTML = `
      <div class="win-bar-wrapper">
        <span class="win-bar-label red-label">Red</span>
        <div class="win-bar-track">
          <div class="win-bar-fill red-fill"></div>
          <div class="win-bar-fill black-fill"></div>
          <div class="win-bar-center"></div>
        </div>
        <span class="win-bar-label black-label">Black</span>
      </div>
      <div class="win-bar-percentage"></div>
    `;

    this.redFill = this.container.querySelector('.red-fill');
    this.blackFill = this.container.querySelector('.black-fill');
    this.percentageEl = this.container.querySelector('.win-bar-percentage');

    // Start hidden
    this.container.style.display = 'none';
  }

  _injectStyles() {
    // Only inject once
    if (document.getElementById('win-bar-styles')) return;

    const style = document.createElement('style');
    style.id = 'win-bar-styles';
    style.textContent = `
      #win-bar {
        margin: 10px 0;
        padding: 8px;
        background: rgba(0, 0, 0, 0.3);
        border-radius: 8px;
      }

      .win-bar-wrapper {
        display: flex;
        align-items: center;
        gap: 8px;
      }

      .win-bar-track {
        flex: 1;
        height: 20px;
        background: #333;
        border-radius: 4px;
        position: relative;
        overflow: hidden;
      }

      .win-bar-fill {
        position: absolute;
        top: 0;
        height: 100%;
        transition: width 0.3s ease;
      }

      .red-fill {
        left: 0;
        background: linear-gradient(to right, #c44, #e55);
        width: 50%;
      }

      .black-fill {
        right: 0;
        background: linear-gradient(to left, #444, #666);
        width: 50%;
      }

      .win-bar-center {
        position: absolute;
        left: 50%;
        top: 0;
        width: 2px;
        height: 100%;
        background: rgba(255, 255, 255, 0.5);
        transform: translateX(-50%);
        z-index: 1;
      }

      .win-bar-label {
        font-size: 12px;
        font-weight: bold;
        width: 40px;
        text-align: center;
      }

      .red-label { color: #e55; }
      .black-label { color: #888; }

      .win-bar-percentage {
        text-align: center;
        font-size: 11px;
        color: #999;
        margin-top: 4px;
      }

      /* Indicator for whose turn it is */
      .win-bar-label.active {
        text-decoration: underline;
      }
    `;
    document.head.appendChild(style);
  }

  /**
   * Update the bar with a value already in Red's perspective.
   * @param {number} valueRed - Value from Red's perspective (-1 to +1)
   */
  updateRed(valueRed) {
    if (!this.container || valueRed === null || valueRed === undefined) return;

    // Clamp for safety
    const redValue = Math.max(-1, Math.min(1, valueRed));
    this.value = redValue;

    const redPercent = ((redValue + 1) / 2) * 100;
    const blackPercent = 100 - redPercent;

    if (this.redFill) this.redFill.style.width = `${redPercent}%`;
    if (this.blackFill) this.blackFill.style.width = `${blackPercent}%`;

    if (this.percentageEl) {
      this.percentageEl.textContent = `Red: ${redPercent.toFixed(0)}%`;
    }
  }

  /**
   * Update the bar with a new evaluation.
   * @deprecated Use updateRed() with server-computed Red-perspective value instead.
   *
   * @param {number} value - NN value from current player's perspective (-1 to +1)
   * @param {string} toMove - Current player ('red' or 'black')
   */
  update(value, toMove) {
    if (!this.container || value === null || value === undefined) return;

    this.toMove = toMove;

    // Convert to red's perspective if needed
    // NN returns value from current player's perspective
    const redValue = toMove === 'red' ? value : -value;
    this.value = redValue;

    // Calculate fill percentages
    // redValue = 1 means red is winning -> red fill = 100%, black = 0%
    // redValue = -1 means black is winning -> red fill = 0%, black = 100%
    // redValue = 0 means even -> both = 50%
    const redPercent = ((redValue + 1) / 2) * 100; // 0% to 100%
    const blackPercent = 100 - redPercent;

    if (this.redFill) this.redFill.style.width = `${redPercent}%`;
    if (this.blackFill) this.blackFill.style.width = `${blackPercent}%`;

    // Show win percentage
    if (this.percentageEl) {
      const redWinPct = redPercent.toFixed(0);
      this.percentageEl.textContent = `Red: ${redWinPct}%`;
    }

    // Highlight whose turn it is
    const redLabel = this.container.querySelector('.red-label');
    const blackLabel = this.container.querySelector('.black-label');
    if (redLabel && blackLabel) {
      redLabel.classList.toggle('active', toMove === 'red');
      blackLabel.classList.toggle('active', toMove === 'black');
    }
  }

  /**
   * Enable/disable the win bar.
   */
  setEnabled(enabled) {
    this.enabled = enabled;
    if (this.container) {
      this.container.style.display = enabled ? 'block' : 'none';
    }
  }

  /**
   * Clear the bar (unknown/initial position).
   */
  clear() {
    this.value = 0;
    if (this.redFill) this.redFill.style.width = '50%';
    if (this.blackFill) this.blackFill.style.width = '50%';
    if (this.percentageEl) this.percentageEl.textContent = '';
  }

  /**
   * Check if the win bar is enabled.
   */
  isEnabled() {
    return this.enabled;
  }
}

// Singleton instance
export const winBar = new WinBar();

export default WinBar;
