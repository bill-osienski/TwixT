/**
 * Zobrist hashing for TwixT game states.
 *
 * Provides O(1) incremental board hash updates via XOR.
 * Used for collision-free caching of game state evaluations.
 */

// Generate a 64-bit random BigInt
function rand64() {
  const hi = BigInt(Math.floor(Math.random() * 2 ** 32));
  const lo = BigInt(Math.floor(Math.random() * 2 ** 32));
  return (hi << 32n) ^ lo;
}

/**
 * Create a Zobrist table for a board of given dimensions.
 * Returns table[row][col][playerIndex] where playerIndex is 0=red, 1=black.
 */
export function makeZobristTable(rows, cols) {
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => [rand64(), rand64()])
  );
}

/**
 * Convert player string to index.
 * @param {string} player - 'red' or 'black'
 * @returns {number} 0 for red, 1 for black
 */
export function playerIndex(player) {
  return player === 'red' ? 0 : 1;
}

// Singleton table for standard 24x24 TwixT board
let _zobristTable = null;

/**
 * Get the shared Zobrist table (lazily initialized).
 * @param {number} size - Board size (default 24)
 * @returns {BigInt[][][]} The Zobrist table
 */
export function getZobristTable(size = 24) {
  if (!_zobristTable || _zobristTable.length !== size) {
    _zobristTable = makeZobristTable(size, size);
  }
  return _zobristTable;
}
