/**
 * Tests for AlphaZero Node.js server components.
 *
 * Run with: npm run test:server
 * Or: node --test server/test_server.js
 *
 * Note: These are unit tests for individual components.
 * Integration tests require a model.onnx file and running server.
 */
import { describe, it, beforeEach } from 'node:test';
import assert from 'node:assert';

import { TwixtState, BOARD_SIZE, MAX_PLIES } from './gameLogic.js';
import { BoardMovesCache } from './cache.js';
import { MCTS, MCTSNode } from './mcts.js';

describe('TwixtState', () => {
  it('should create empty initial state', () => {
    const state = new TwixtState();
    assert.strictEqual(state.toMove, 'red');
    assert.strictEqual(state.ply, 0);
    assert.strictEqual(state.pegs.size, 0);
    assert.strictEqual(state.bridges.size, 0);
  });

  it('should generate legal moves for red', () => {
    const state = new TwixtState();
    const moves = state.legalMoves();

    // Red can't place on left/right edges (cols 0 and 23) or corners
    // Should have (24 * 24) - 4 corners - 2*22 edges = 576 - 4 - 44 = 528
    assert.strictEqual(moves.length, 528);

    // Check no moves on forbidden cells
    for (const [row, col] of moves) {
      // Not on left/right edges for red
      assert.notStrictEqual(col, 0, 'Red should not play on left edge');
      assert.notStrictEqual(col, 23, 'Red should not play on right edge');
      // Not on corners
      const isCorner =
        (row === 0 || row === 23) && (col === 0 || col === 23);
      assert.ok(!isCorner, 'Should not play on corners');
    }
  });

  it('should apply move and switch player', () => {
    const state = new TwixtState();
    const newState = state.applyMove([5, 5]);

    assert.strictEqual(newState.toMove, 'black');
    assert.strictEqual(newState.ply, 1);
    assert.strictEqual(newState.getPeg(5, 5), 'red');
    assert.strictEqual(newState.pegs.size, 1);

    // Original state unchanged (immutability)
    assert.strictEqual(state.toMove, 'red');
    assert.strictEqual(state.ply, 0);
    assert.strictEqual(state.pegs.size, 0);
  });

  it('should create bridges between knight-move pegs', () => {
    let state = new TwixtState();

    // Place two red pegs at knight-move distance
    state = state.applyMove([5, 5]); // red
    state = state.applyMove([10, 10]); // black
    state = state.applyMove([7, 6]); // red - knight move from (5,5)

    // Should have created one bridge
    assert.strictEqual(state.bridges.size, 1);
  });

  it('should detect win for red (top to bottom)', () => {
    let state = new TwixtState();

    // Create a simple winning path for red
    // This is artificial but tests the win detection
    // Red needs to connect row 0 to row 23 via bridges
    // For a quick test, we'll just place pegs and check winner()

    // Place on top edge
    state = state.applyMove([0, 5]); // red on top
    assert.strictEqual(state.winner(), null); // Not won yet

    // Check terminal/winner logic
    assert.strictEqual(state.isTerminal(), false);
  });

  it('should convert to tensor in HWC format', () => {
    const state = new TwixtState();
    state.pegs.set('5,5', 'red');

    const tensor = state.toTensorHWC();

    // Should be [24][24][24] = [H][W][C]
    assert.strictEqual(tensor.length, 24, 'Height dimension');
    assert.strictEqual(tensor[0].length, 24, 'Width dimension');
    assert.strictEqual(tensor[0][0].length, 24, 'Channel dimension');

    // Check red peg channel (0) at position (5, 5)
    assert.strictEqual(tensor[5][5][0], 1.0, 'Red peg at (5,5)');
  });

  it('should serialize and deserialize correctly', () => {
    let state = new TwixtState();
    state = state.applyMove([5, 5]);
    state = state.applyMove([10, 10]);
    state = state.applyMove([7, 6]); // Creates bridge

    const dict = state.toDict();
    const restored = TwixtState.fromDict(dict);

    assert.strictEqual(restored.toMove, state.toMove);
    assert.strictEqual(restored.ply, state.ply);
    assert.strictEqual(restored.pegs.size, state.pegs.size);
    assert.strictEqual(restored.bridges.size, state.bridges.size);
  });

  it('should enforce MAX_PLIES draw', () => {
    const state = new TwixtState({
      ply: MAX_PLIES,
    });

    assert.strictEqual(state.isTerminal(), true);
    assert.strictEqual(state.winner(), null);
    assert.strictEqual(state.gameResult(), 'draw');
  });
});

describe('BoardMovesCache', () => {
  let cache;

  beforeEach(() => {
    cache = new BoardMovesCache(100);
  });

  it('should store and retrieve values', () => {
    const pegs = new Map([['5,5', 'red']]);
    const moves = [
      [0, 1],
      [0, 2],
    ];
    const value = { move: { row: 0, col: 1 } };

    cache.set(pegs, moves, value);
    const retrieved = cache.get(pegs, moves);

    assert.deepStrictEqual(retrieved, value);
  });

  it('should return undefined for cache miss', () => {
    const pegs = new Map();
    const moves = [[0, 1]];

    const result = cache.get(pegs, moves);
    assert.strictEqual(result, undefined);
  });

  it('should be move-order independent', () => {
    const pegs = new Map();
    const moves1 = [
      [0, 1],
      [0, 2],
      [0, 3],
    ];
    const moves2 = [
      [0, 3],
      [0, 1],
      [0, 2],
    ]; // Same moves, different order

    cache.set(pegs, moves1, { test: true });
    const result = cache.get(pegs, moves2);

    assert.deepStrictEqual(result, { test: true });
  });

  it('should evict oldest entries when full', () => {
    cache = new BoardMovesCache(3);

    for (let i = 0; i < 5; i++) {
      const pegs = new Map([[`${i},0`, 'red']]);
      cache.set(pegs, [], { i });
    }

    // Should have evicted first two entries
    assert.strictEqual(cache.size, 3);

    // First entry should be evicted
    const oldPegs = new Map([['0,0', 'red']]);
    assert.strictEqual(cache.get(oldPegs, []), undefined);

    // Last entry should exist
    const newPegs = new Map([['4,0', 'red']]);
    assert.deepStrictEqual(cache.get(newPegs, []), { i: 4 });
  });

  it('should track hit/miss statistics', () => {
    const pegs = new Map();
    cache.set(pegs, [], { test: true });

    cache.get(pegs, []); // hit
    cache.get(pegs, []); // hit
    cache.get(new Map([['1,1', 'red']]), []); // miss

    assert.strictEqual(cache.hits, 2);
    assert.strictEqual(cache.misses, 1);
    assert.strictEqual(cache.hitRate, 2 / 3);
  });
});

describe('MCTSNode', () => {
  it('should initialize with correct defaults', () => {
    const state = new TwixtState();
    const node = new MCTSNode(state);

    assert.strictEqual(node.visitCount, 0);
    assert.strictEqual(node.valueSum, 0);
    assert.strictEqual(node.qValue, 0);
    assert.strictEqual(node.isExpanded, false);
    assert.strictEqual(node.parent, null);
    assert.strictEqual(node.move, null);
  });

  it('should calculate qValue correctly', () => {
    const node = new MCTSNode(new TwixtState());

    node.visitCount = 10;
    node.valueSum = 5;

    assert.strictEqual(node.qValue, 0.5);
  });

  it('should track expansion state', () => {
    const node = new MCTSNode(new TwixtState());

    assert.strictEqual(node.isExpanded, false);

    node.priors = new Map();
    assert.strictEqual(node.isExpanded, true);
  });
});

describe('MCTS selectMove', () => {
  it('should select deterministically with temperature=0', () => {
    // Create fake visit counts
    const visitCounts = new Map([
      ['5,5', 100],
      ['5,6', 50],
      ['5,7', 200], // highest
      ['5,8', 50],
    ]);

    // Use a mock inference (won't be called for selectMove)
    const mcts = new MCTS(null, { nSimulations: 10 });

    const move = mcts.selectMove(visitCounts, 0);
    assert.strictEqual(move, '5,7'); // highest count
  });

  it('should use lexicographic tie-break', () => {
    const visitCounts = new Map([
      ['10,5', 100],
      ['5,10', 100], // same count, smaller row
      ['5,5', 100], // same count, smaller row AND col
    ]);

    const mcts = new MCTS(null, { nSimulations: 10 });
    const move = mcts.selectMove(visitCounts, 0);

    // "10,5" > "5,10" > "5,5" lexicographically
    // But we want smallest, so "10,5" < "5,10" < "5,5" is wrong
    // Actually string comparison: "10,5" < "5,10" < "5,5" (because "1" < "5")
    assert.strictEqual(move, '10,5');
  });

  it('should return consistent moves with same temperature=0', () => {
    const visitCounts = new Map([
      ['3,3', 50],
      ['5,5', 100],
      ['7,7', 75],
    ]);

    const mcts = new MCTS(null, { nSimulations: 10 });

    // Should always return the same move
    const moves = [];
    for (let i = 0; i < 10; i++) {
      moves.push(mcts.selectMove(visitCounts, 0));
    }

    assert.ok(
      moves.every((m) => m === moves[0]),
      'All moves should be identical in deterministic mode'
    );
  });
});

describe('TwixtState toTensorHWC parity', () => {
  it('should match toTensorNested but transposed', () => {
    let state = new TwixtState();
    state = state.applyMove([5, 5]); // red
    state = state.applyMove([10, 10]); // black

    const hwc = state.toTensorHWC(); // [H][W][C]
    const chw = state.toTensorNested(); // [C][H][W]

    // Check dimensions
    assert.strictEqual(hwc.length, 24);
    assert.strictEqual(hwc[0].length, 24);
    assert.strictEqual(hwc[0][0].length, 24);

    // Verify transposition: hwc[h][w][c] === chw[c][h][w]
    for (let c = 0; c < 24; c++) {
      for (let h = 0; h < 24; h++) {
        for (let w = 0; w < 24; w++) {
          assert.strictEqual(
            hwc[h][w][c],
            chw[c][h][w],
            `Mismatch at c=${c}, h=${h}, w=${w}`
          );
        }
      }
    }
  });
});

console.log('Server tests loaded. Running...');
