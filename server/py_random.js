/**
 * A transcription of the subset of CPython's `random.Random` that the H-TB
 * protocol search uses (integration proposal §4.3): integer seeding, the
 * MT19937 generator, getrandbits, _randbelow, random(), choice and shuffle.
 *
 * Why a port rather than any generator: with the same seed it reproduces the
 * research search's streams exactly, so the JS port of the search can be tested
 * against the Python search seed for seed, including its random tie-breaks
 * (server/test_py_random.js, server/test_protocol_mcts.js). In production each
 * request seeds fresh streams (server/protocol_move.js), so the choice of
 * generator does not change any distribution.
 *
 * Deliberately limited to these operations. Pinned against the .venv's CPython
 * (3.14.10) by the fixtures in tests/fixtures/htb_integration/.
 */

const N = 624;
const M = 397;
const MATRIX_A = 0x9908b0df;
const UPPER_MASK = 0x80000000;
const LOWER_MASK = 0x7fffffff;

/** Number of bits needed to represent a positive integer (Python's int.bit_length). */
function bitLength(n) {
  let k = 0;
  while (n > 0) {
    k += 1;
    n = Math.floor(n / 2);
  }
  return k;
}

export class PyRandom {
  /**
   * @param {number} seed - a non-negative safe integer, as `random.Random(seed)`
   */
  constructor(seed) {
    if (!Number.isSafeInteger(seed) || seed < 0) {
      throw new TypeError(`PyRandom seed must be a non-negative safe integer, got ${seed}`);
    }
    this.mt = new Uint32Array(N);
    this.mti = N + 1;
    // CPython random_seed: the absolute value's 32-bit words, least significant
    // first; zero is the one-word key [0].
    const key = [];
    let rest = seed;
    do {
      key.push(rest % 0x100000000);
      rest = Math.floor(rest / 0x100000000);
    } while (rest > 0);
    this._initByArray(key);
  }

  _initGenrand(s) {
    const mt = this.mt;
    mt[0] = s >>> 0;
    for (let i = 1; i < N; i++) {
      const prev = mt[i - 1] ^ (mt[i - 1] >>> 30);
      mt[i] = (Math.imul(1812433253, prev) + i) >>> 0;
    }
    this.mti = N;
  }

  _initByArray(key) {
    const mt = this.mt;
    this._initGenrand(19650218);
    let i = 1;
    let j = 0;
    for (let k = Math.max(N, key.length); k > 0; k--) {
      const s = mt[i - 1] ^ (mt[i - 1] >>> 30);
      mt[i] = ((mt[i] ^ Math.imul(s, 1664525)) + key[j] + j) >>> 0;
      i++;
      j++;
      if (i >= N) {
        mt[0] = mt[N - 1];
        i = 1;
      }
      if (j >= key.length) j = 0;
    }
    for (let k = N - 1; k > 0; k--) {
      const s = mt[i - 1] ^ (mt[i - 1] >>> 30);
      mt[i] = ((mt[i] ^ Math.imul(s, 1566083941)) - i) >>> 0;
      i++;
      if (i >= N) {
        mt[0] = mt[N - 1];
        i = 1;
      }
    }
    mt[0] = 0x80000000;
  }

  /** genrand_uint32: one tempered 32-bit output. */
  _next() {
    const mt = this.mt;
    if (this.mti >= N) {
      let kk = 0;
      let y;
      for (; kk < N - M; kk++) {
        y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
        mt[kk] = mt[kk + M] ^ (y >>> 1) ^ (y & 1 ? MATRIX_A : 0);
      }
      for (; kk < N - 1; kk++) {
        y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
        mt[kk] = mt[kk + (M - N)] ^ (y >>> 1) ^ (y & 1 ? MATRIX_A : 0);
      }
      y = (mt[N - 1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
      mt[N - 1] = mt[M - 1] ^ (y >>> 1) ^ (y & 1 ? MATRIX_A : 0);
      this.mti = 0;
    }
    let y = mt[this.mti++];
    y ^= y >>> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >>> 18;
    return y >>> 0;
  }

  /** random(): a float in [0, 1) with 53 random bits. */
  random() {
    const a = this._next() >>> 5;
    const b = this._next() >>> 6;
    return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
  }

  /** getrandbits(k) for 0 <= k <= 32, the only widths this search needs. */
  getrandbits(k) {
    if (!Number.isInteger(k) || k < 0 || k > 32) {
      throw new RangeError(`getrandbits supports 0..32 bits here, got ${k}`);
    }
    if (k === 0) return 0;
    return this._next() >>> (32 - k);
  }

  /** _randbelow_with_getrandbits(n): uniform in [0, n), n >= 1. */
  _randbelow(n) {
    const k = bitLength(n);
    let r = this.getrandbits(k);
    while (r >= n) r = this.getrandbits(k);
    return r;
  }

  /** choice(seq) */
  choice(seq) {
    if (seq.length === 0) throw new RangeError('Cannot choose from an empty sequence');
    return seq[this._randbelow(seq.length)];
  }

  /** shuffle(x), in place: Fisher-Yates from the end, as CPython. */
  shuffle(x) {
    for (let i = x.length - 1; i >= 1; i--) {
      const j = this._randbelow(i + 1);
      const t = x[i];
      x[i] = x[j];
      x[j] = t;
    }
  }
}
