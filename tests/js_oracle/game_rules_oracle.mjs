#!/usr/bin/env node
/**
 * Node.js oracle for game rules parity testing.
 *
 * Reads moves from stdin (JSON format), applies them, and outputs state.
 *
 * Input format:
 *   {"moves": [[row, col], ...], "include_tensor": boolean}
 *
 * Output format:
 *   {
 *     "legal_moves": [[row, col], ...],
 *     "is_terminal": boolean,
 *     "winner": "red"|"black"|null,
 *     "to_move": "red"|"black",
 *     "ply": number,
 *     "tensor": [[[...]]] // Only if include_tensor is true
 *   }
 */

import { TwixtState } from '../../server/gameLogic.js';

async function readStdin() {
  const chunks = [];
  for await (const chunk of process.stdin) {
    chunks.push(chunk);
  }
  return Buffer.concat(chunks).toString('utf8');
}

async function main() {
  try {
    const input = await readStdin();
    const data = JSON.parse(input);
    const moves = data.moves || [];
    const includeTensor = data.include_tensor || false;

    // Apply moves
    const state = TwixtState.fromMoves(moves);

    // Get state info
    const result = {
      legal_moves: state.legalMoves(),
      is_terminal: state.isTerminal(),
      winner: state.winner(),
      to_move: state.toMove,
      ply: state.ply,
    };

    // Include tensor if requested
    if (includeTensor) {
      result.tensor = state.toTensorNested();
    }

    console.log(JSON.stringify(result));
  } catch (err) {
    console.error('Oracle error:', err.message);
    console.error(err.stack);
    process.exit(1);
  }
}

main();
