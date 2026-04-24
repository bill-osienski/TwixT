/**
 * ONNX model wrapper for AlphaZero inference.
 *
 * Handles tensor format conversion between JS and ONNX:
 * - Input: toTensorHWC() returns [H][W][C] format
 * - ONNX: Expects NCHW format (1, C, H, W)
 * - Output: Policy logits and value scalar
 *
 * This is the ONLY place Node.js does layout conversion.
 */
import * as ort from 'onnxruntime-node';
import { NUM_CHANNELS } from './gameLogic.js';

export class AlphaZeroInference {
  constructor(modelPath) {
    this.modelPath = modelPath;
    this.session = null;
    this.maxMoves = 512;
  }

  async load() {
    this.session = await ort.InferenceSession.create(this.modelPath);
    console.log(`ONNX model loaded: ${this.modelPath}`);
    console.log(`  Inputs: ${this.session.inputNames.join(', ')}`);
    console.log(`  Outputs: ${this.session.outputNames.join(', ')}`);
  }

  async evaluate(boardTensorHWC, moves) {
    /**
     * Evaluate position with neural network.
     *
     * Args:
     *   boardTensorHWC: [24][24][24] board representation in [H][W][C] format
     *                   from TwixtState.toTensorHWC()
     *   moves: Array of [row, col] legal moves
     *
     * Returns:
     *   { priors: Map<string, number>, value: number }
     *   priors maps "row,col" -> probability (after softmax)
     *   value in [-1, 1]
     */
    if (!this.session) {
      throw new Error('Model not loaded. Call load() first.');
    }

    // Keep input tensor shape in sync with gameLogic.js::NUM_CHANNELS.
    // Post-connectivity-retrain (2026-04-19) this is 30. Hardcoding would
    // produce [1, 24, 24, 24] input, ONNX would reject with a shape
    // mismatch, and the evaluate/analyze-position handlers would return
    // 500 — which is exactly how this drift was caught.
    const numChannels = NUM_CHANNELS;
    const size = 24;

    // CRITICAL: Convert from toTensorHWC() [H][W][C] to ONNX NCHW (1, C, H, W)
    // This is the ONLY place we do layout conversion
    const board = new Float32Array(1 * numChannels * size * size);
    for (let c = 0; c < numChannels; c++) {
      for (let r = 0; r < size; r++) {
        for (let col = 0; col < size; col++) {
          board[c * size * size + r * size + col] = boardTensorHWC[r][col][c];
        }
      }
    }

    // Prepare move arrays (padded to maxMoves=512)
    const moveRows = new BigInt64Array(this.maxMoves);
    const moveCols = new BigInt64Array(this.maxMoves);
    const moveMask = new Float32Array(this.maxMoves);

    for (let i = 0; i < moves.length && i < this.maxMoves; i++) {
      // moves are [row, col] arrays
      moveRows[i] = BigInt(moves[i][0]);
      moveCols[i] = BigInt(moves[i][1]);
      moveMask[i] = 1.0;
    }
    // Padding is already 0 (BigInt default is 0n, Float32 default is 0.0)

    // Run inference
    const feeds = {
      board: new ort.Tensor('float32', board, [1, numChannels, size, size]),
      move_rows: new ort.Tensor('int64', moveRows, [this.maxMoves]),
      move_cols: new ort.Tensor('int64', moveCols, [this.maxMoves]),
      move_mask: new ort.Tensor('float32', moveMask, [this.maxMoves]),
    };

    const results = await this.session.run(feeds);

    // Extract logits (only for valid moves, first moves.length entries)
    const allLogits = results.policy_logits.data;
    const logits = [];
    for (let i = 0; i < moves.length; i++) {
      logits.push(allLogits[i]);
    }

    let value = results.value.data[0];

    // Safety: if value escapes [-1,1], it's definitely pretanh
    if (value < -1.0 || value > 1.0) value = Math.tanh(value);

    // Clamp to valid range
    value = Math.max(-1, Math.min(1, value));

    // Softmax for priors (used in MCTS) - loop max, no spread operator
    let maxLogit = -Infinity;
    for (let i = 0; i < logits.length; i++) {
      if (logits[i] > maxLogit) maxLogit = logits[i];
    }

    let sumExp = 0;
    const exps = [];
    for (let i = 0; i < logits.length; i++) {
      const exp = Math.exp(logits[i] - maxLogit);
      exps.push(exp);
      sumExp += exp;
    }

    const priors = new Map();
    for (let i = 0; i < moves.length; i++) {
      const key = `${moves[i][0]},${moves[i][1]}`;
      priors.set(key, exps[i] / sumExp);
    }

    return { priors, value };
  }

  /**
   * Get raw logits without softmax (for debugging/testing).
   */
  async evaluateRaw(boardTensorHWC, moves) {
    if (!this.session) {
      throw new Error('Model not loaded. Call load() first.');
    }

    // Keep input tensor shape in sync with gameLogic.js::NUM_CHANNELS.
    // Post-connectivity-retrain (2026-04-19) this is 30. Hardcoding would
    // produce [1, 24, 24, 24] input, ONNX would reject with a shape
    // mismatch, and the evaluate/analyze-position handlers would return
    // 500 — which is exactly how this drift was caught.
    const numChannels = NUM_CHANNELS;
    const size = 24;

    const board = new Float32Array(1 * numChannels * size * size);
    for (let c = 0; c < numChannels; c++) {
      for (let r = 0; r < size; r++) {
        for (let col = 0; col < size; col++) {
          board[c * size * size + r * size + col] = boardTensorHWC[r][col][c];
        }
      }
    }

    const moveRows = new BigInt64Array(this.maxMoves);
    const moveCols = new BigInt64Array(this.maxMoves);
    const moveMask = new Float32Array(this.maxMoves);

    for (let i = 0; i < moves.length && i < this.maxMoves; i++) {
      moveRows[i] = BigInt(moves[i][0]);
      moveCols[i] = BigInt(moves[i][1]);
      moveMask[i] = 1.0;
    }

    const feeds = {
      board: new ort.Tensor('float32', board, [1, numChannels, size, size]),
      move_rows: new ort.Tensor('int64', moveRows, [this.maxMoves]),
      move_cols: new ort.Tensor('int64', moveCols, [this.maxMoves]),
      move_mask: new ort.Tensor('float32', moveMask, [this.maxMoves]),
    };

    const results = await this.session.run(feeds);

    const logits = [];
    for (let i = 0; i < moves.length; i++) {
      logits.push(results.policy_logits.data[i]);
    }

    return {
      logits,
      value: results.value.data[0],
    };
  }
}
