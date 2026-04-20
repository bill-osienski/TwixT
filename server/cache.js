/**
 * Position cache with board+moves hash key.
 *
 * Uses:
 * - Uint8Array for board representation (fast)
 * - FNV-1a hashing (fast, good distribution)
 * - LRU eviction (Map iteration order)
 * - Sorted moves for order-independent cache hits
 *
 * Optionally supports "value-only by board" caching when move-set is implied.
 */

// FNV-1a constants (32-bit)
const FNV_OFFSET = 2166136261;
const FNV_PRIME = 16777619;

export class BoardMovesCache {
  constructor(maxSize = 10000) {
    this.cache = new Map();
    this.maxSize = maxSize;
    this.hits = 0;
    this.misses = 0;
  }

  /**
   * Convert board pegs to Uint8Array for fast hashing.
   * 0 = empty, 1 = red, 2 = black
   * @param {Map<string, string>} pegs - Map of "r,c" -> player
   * @param {number} size - Board size
   */
  _pegsToUint8(pegs, size = 24) {
    const arr = new Uint8Array(size * size);
    for (const [key, player] of pegs) {
      const [r, c] = key.split(',').map(Number);
      const idx = r * size + c;
      arr[idx] = player === 'red' ? 1 : 2;
    }
    return arr;
  }

  /**
   * FNV-1a hash for Uint8Array (fast loop, no spread).
   */
  _fnv1a(data) {
    let hash = FNV_OFFSET;
    for (let i = 0; i < data.length; i++) {
      hash ^= data[i];
      hash = Math.imul(hash, FNV_PRIME);
    }
    return hash >>> 0; // Convert to unsigned
  }

  _hashPegs(pegs, size = 24) {
    const arr = this._pegsToUint8(pegs, size);
    return this._fnv1a(arr);
  }

  _hashMoves(moves) {
    // CRITICAL: Sort moves for order-independent hashing
    // moves are [row, col] arrays
    const sorted = [...moves].sort((a, b) => a[0] - b[0] || a[1] - b[1]);

    // Pack moves into Uint8Array (2 bytes per move: row, col)
    const arr = new Uint8Array(sorted.length * 2);
    for (let i = 0; i < sorted.length; i++) {
      arr[i * 2] = sorted[i][0];
      arr[i * 2 + 1] = sorted[i][1];
    }
    return this._fnv1a(arr);
  }

  makeKey(pegs, moves, size = 24) {
    const pegsHash = this._hashPegs(pegs, size);
    const movesHash = this._hashMoves(moves);
    return `${pegsHash}:${movesHash}`;
  }

  /**
   * Key for value-only caching (when move-set is implied by board).
   */
  makeBoardOnlyKey(pegs, size = 24) {
    return `v:${this._hashPegs(pegs, size)}`;
  }

  get(pegs, moves, size = 24) {
    const key = this.makeKey(pegs, moves, size);
    const value = this.cache.get(key);

    // LRU: move to end on access
    if (value !== undefined) {
      this.cache.delete(key);
      this.cache.set(key, value);
      this.hits++;
    } else {
      this.misses++;
    }
    return value;
  }

  set(pegs, moves, value, size = 24) {
    const key = this.makeKey(pegs, moves, size);

    // If key exists, delete first (for LRU ordering)
    if (this.cache.has(key)) {
      this.cache.delete(key);
    }

    // LRU eviction: remove oldest (first) entry
    if (this.cache.size >= this.maxSize) {
      const firstKey = this.cache.keys().next().value;
      this.cache.delete(firstKey);
    }

    this.cache.set(key, value);
  }

  clear() {
    this.cache.clear();
    this.hits = 0;
    this.misses = 0;
  }

  get hitRate() {
    const total = this.hits + this.misses;
    return total === 0 ? 0 : this.hits / total;
  }

  get size() {
    return this.cache.size;
  }
}
