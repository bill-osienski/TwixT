/**
 * The JS twin of generate_fixtures.py's FakeEvaluator: integer arithmetic, one
 * true division and float32 rounding, so it reproduces the Python fake bit for
 * bit. No model. Used by the H-TB integration's G1 tests.
 */
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

export const FIXTURE_DIR = dirname(fileURLToPath(import.meta.url));
export const REPO_ROOT = join(FIXTURE_DIR, '..', '..', '..');

export const loadFixture = (name) => JSON.parse(readFileSync(join(FIXTURE_DIR, name), 'utf8'));

/** The research sources a fixture was generated from must be byte-identical today. */
export function staleSources(fixture) {
  const stale = [];
  for (const [rel, sha] of Object.entries(fixture.provenance.sources_sha256)) {
    const now = createHash('sha256').update(readFileSync(join(REPO_ROOT, rel))).digest('hex');
    if (now !== sha) stale.push(rel);
  }
  return stale;
}

export function boardHash(hwc) {
  let h = 0;
  for (let r = 0; r < 24; r++) {
    for (let c = 0; c < 24; c++) {
      const idx = r * 24 + c + 1;
      const ch = hwc[r][c];
      if (ch[0] > 0.5) h += idx * 3;
      if (ch[1] > 0.5) h += idx * 5;
      for (let k = 2; k < 18; k++) if (ch[k] > 0.5) h += idx * (k + 7);
    }
  }
  return h % 100003;
}

export const fakeInference = () => ({
  calls: 0,
  async evaluate(hwc, moves) {
    this.calls += 1;
    const h = boardHash(hwc);
    const w = moves.map(([r, c]) => ((r * 31 + c * 17 + h) % 97) + 1);
    let s = 0;
    for (const x of w) s += x;
    const priors = new Map();
    for (let j = 0; j < moves.length; j++) {
      priors.set(`${moves[j][0]},${moves[j][1]}`, Math.fround(w[j] / s + (j % 3) * 4e-9));
    }
    const value = Math.fround((((h * 37 + moves.length) % 201) - 100) / 128);
    return { priors, value };
  },
});

/** sha256 of an [H][W][C] tensor as little-endian float32, as numpy's tobytes(). */
export function hwcSha256(hwc) {
  const flat = new Float32Array(24 * 24 * hwc[0][0].length);
  let i = 0;
  for (const row of hwc) for (const cell of row) for (const v of cell) flat[i++] = v;
  return createHash('sha256').update(Buffer.from(flat.buffer)).digest('hex');
}

export const f32Sha256 = (values) =>
  createHash('sha256').update(Buffer.from(new Float32Array(values).buffer)).digest('hex');

/** Python's json.dumps of a list of strings (", " separator). */
export const pyJsonList = (xs) => `[${xs.map((x) => JSON.stringify(x)).join(', ')}]`;
