import assert from 'node:assert/strict';

// Ensure the core AI module loads without throwing and exposes the TwixTAI class.
const { default: TwixTAI } = await import('../assets/js/ai/search.js');
assert.equal(typeof TwixTAI, 'function', 'TwixTAI should be a constructor');

// Heuristic helpers should evaluate simple situations without crashing.
const heuristics = await import('../assets/js/ai/heuristics.js');
assert.equal(
  typeof heuristics.evaluatePosition,
  'function',
  'evaluatePosition should be available'
);

const { default: TwixTGame } = await import('../assets/js/game/twixtGame.js');
const gameInstance = new TwixTGame();
const ai = new TwixTAI(gameInstance, 'red');
assert.ok(ai, 'AI should instantiate on game instance');
