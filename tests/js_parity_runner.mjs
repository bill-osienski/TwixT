#!/usr/bin/env node
/** Minimal JS-side tensor constructor for parity tests.
 *
 * Accepts --active-size and --moves (JSON array of [r,c] tuples) and emits
 * a flattened JSON array of the (NUM_CHANNELS * boardSize * boardSize)
 * tensor values in channel-major order.
 */
import { parseArgs } from 'node:util';
import { buildStateTensor } from '../server/gameLogic.js';

const { values } = parseArgs({
  options: {
    'active-size': { type: 'string' },
    moves: { type: 'string' },
  },
});

const active = parseInt(values['active-size'], 10);
const moves = JSON.parse(values['moves']);
const tensor = buildStateTensor(active, moves);

// tensor is array of Float32Array (one per channel). Flatten to single array.
const flat = [];
for (const channel of tensor) {
  for (const v of channel) {
    flat.push(v);
  }
}
process.stdout.write(JSON.stringify(flat));
