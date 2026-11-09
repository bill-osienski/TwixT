/**
 * V1: the mirror transform and mirror-averaged outputs, exactly equal to the
 * research's (_mirror_position_lr, MirrorAveragingEvaluator), over the Phase 2
 * parity corpus, with a fake inner evaluator. No model.
 */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { test } from 'node:test';

import { TwixtState } from './gameLogic.js';
import {
  MIRROR_DIR_PERM,
  MirrorAveragingInference,
  mirrorBoardHWC,
  mirrorMoves,
} from './mirror_inference.js';
import {
  f32Sha256,
  fakeInference,
  hwcSha256,
  loadFixture,
  pyJsonList,
  staleSources,
} from '../tests/fixtures/htb_integration/fake_evaluator.mjs';

const FIX = loadFixture('mirror.json');
const keyOf = (m) => `${m[0]},${m[1]}`;
const createSha = (text) => createHash('sha256').update(text).digest('hex');

test('the mirror fixture describes the research code as it is today', () => {
  assert.deepEqual(staleSources(FIX), []);
  assert.ok(FIX.positions.length >= 120, `only ${FIX.positions.length} positions`);
});

test('the link-direction permutation maps (dr, dc) to (dr, -dc)', () => {
  assert.deepEqual([...MIRROR_DIR_PERM], [7, 6, 5, 4, 3, 2, 1, 0]);
});

test('every corpus position: input, mirrored board and mirrored moves equal Python exactly', () => {
  let n = 0;
  for (const p of FIX.positions) {
    const s = TwixtState.fromMoves(p.moves);
    const hwc = s.toTensorHWC();
    assert.equal(hwcSha256(hwc), p.board_sha256, `${p.id}: the input tensor differs (S1)`);
    assert.equal(hwcSha256(mirrorBoardHWC(hwc)), p.mirrored_sha256, `${p.id}: mirrored board`);
    const mm = mirrorMoves(s.legalMoves()).map(keyOf);
    assert.equal(
      createSha(pyJsonList(mm)),
      p.mirrored_moves_sha256,
      `${p.id}: mirrored legal moves`
    );
    if (p.mirrored_moves) assert.deepEqual(mm, p.mirrored_moves);
    n += 1;
  }
  assert.equal(n, FIX.positions.length);
});

test('mirroring twice is the identity', () => {
  for (const p of FIX.positions.slice(0, 20)) {
    const hwc = TwixtState.fromMoves(p.moves).toTensorHWC();
    assert.deepEqual(mirrorBoardHWC(mirrorBoardHWC(hwc)), hwc, p.id);
    const moves = TwixtState.fromMoves(p.moves).legalMoves();
    assert.deepEqual(mirrorMoves(mirrorMoves(moves)), moves);
  }
});

test('mirror-averaged priors and value equal MirrorAveragingEvaluator(fake) exactly', async () => {
  const mirror = new MirrorAveragingInference(fakeInference());
  for (const p of FIX.positions) {
    const s = TwixtState.fromMoves(p.moves);
    const moves = s.legalMoves();
    const { priors, value } = await mirror.evaluate(s.toTensorHWC(), moves);
    const arr = moves.map((m) => priors.get(keyOf(m)));
    assert.equal(f32Sha256([...arr, value]), p.averaged_sha256, `${p.id}: averaged outputs`);
    if (p.averaged_priors) {
      assert.deepEqual(arr, p.averaged_priors);
      assert.equal(value, p.averaged_value);
    }
  }
});

test('the averaged comparison can fail: an un-mirrored second evaluation differs', async () => {
  // CONSTRUCTED negative case: averaging the board with ITSELF must not match,
  // or the test above would be vacuous.
  const base = fakeInference();
  const p = FIX.positions[5];
  const s = TwixtState.fromMoves(p.moves);
  const moves = s.legalMoves();
  const a = await base.evaluate(s.toTensorHWC(), moves);
  const arr = moves.map((m) => a.priors.get(keyOf(m)));
  assert.notEqual(f32Sha256([...arr, a.value]), p.averaged_sha256);
});

test('the wrapper calls its base exactly twice per evaluation', async () => {
  const base = fakeInference();
  const mirror = new MirrorAveragingInference(base);
  const s = TwixtState.fromMoves(FIX.positions[0].moves);
  await mirror.evaluate(s.toTensorHWC(), s.legalMoves());
  assert.equal(base.calls, 2);
});
