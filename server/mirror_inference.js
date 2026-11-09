/**
 * Mirror-averaging inference for the H-TB protocol (integration proposal §4.2).
 *
 * A transcription of scripts/GPU/alphazero/mirror_averaging.py
 * (MirrorAveragingEvaluator) and of self_play._mirror_position_lr for the full
 * 24x24 board. Each evaluation runs the wrapped inference twice, on the board
 * and on its left-right mirror, and returns the index-aligned mean of the two
 * prior distributions (no renormalization) and the mean of the two
 * side-to-move values.
 *
 * The research computes the mean in float32 (numpy: `((p + q) / 2).astype(
 * np.float32)` over float32 inputs), so this wrapper rounds through float32 the
 * same way. Used only by the `protocol` search; easy, medium and the analysis
 * endpoints never see it.
 */
import { BOARD_SIZE, NUM_CHANNELS } from './gameLogic.js';

// Restated from server/gameLogic.js (not exported there; that file is not
// edited). V1 (server/test_mirror_inference.js) checks the resulting mirror
// against Python's element for element, which is what pins these values.
const DIRECTION_TO_CHANNEL = {
  '2,1': 0,
  '1,2': 1,
  '-1,2': 2,
  '-2,1': 3,
  '-2,-1': 4,
  '-1,-2': 5,
  '1,-2': 6,
  '2,-1': 7,
};
const CHANNEL_RED_LINKS_START = 2;
const CHANNEL_BLACK_LINKS_START = 10;
const CHANNEL_BLACK_LEFT_DIST = 21;
const CHANNEL_BLACK_RIGHT_DIST = 22;
const CHANNEL_BLACK_CONN_LEFT = 27;
const CHANNEL_BLACK_CONN_RIGHT = 28;

/** perm[src] = dst: link direction (dr, dc) maps to (dr, -dc). As _build_mirror_dir_perm. */
export const MIRROR_DIR_PERM = Object.freeze(
  (() => {
    const perm = new Array(8);
    for (const [key, i] of Object.entries(DIRECTION_TO_CHANNEL)) {
      const [dr, dc] = key.split(',').map(Number);
      const mirrored = `${dr},${-dc}`;
      if (!(mirrored in DIRECTION_TO_CHANNEL)) {
        throw new Error(`missing mirrored direction for ${key}`);
      }
      perm[i] = DIRECTION_TO_CHANNEL[mirrored];
    }
    return perm;
  })()
);

/**
 * The left-right mirror of an [H][W][C] tensor over the full board, as
 * `_mirror_position_lr(board, [], [], 24)[0]`.
 */
export function mirrorBoardHWC(hwc) {
  const S = BOARD_SIZE;
  if (hwc.length !== S || hwc[0].length !== S || hwc[0][0].length !== NUM_CHANNELS) {
    throw new Error(`mirror expects a ${S}x${S}x${NUM_CHANNELS} tensor`);
  }
  const out = new Array(S);
  for (let r = 0; r < S; r++) {
    const row = new Array(S);
    for (let c = 0; c < S; c++) {
      const src = hwc[r][S - 1 - c];
      const ch = src.slice();
      // Link directions, per colour: out[base + perm[i]] = in[base + i].
      for (const base of [CHANNEL_RED_LINKS_START, CHANNEL_BLACK_LINKS_START]) {
        for (let i = 0; i < 8; i++) ch[base + MIRROR_DIR_PERM[i]] = src[base + i];
      }
      ch[CHANNEL_BLACK_LEFT_DIST] = src[CHANNEL_BLACK_RIGHT_DIST];
      ch[CHANNEL_BLACK_RIGHT_DIST] = src[CHANNEL_BLACK_LEFT_DIST];
      ch[CHANNEL_BLACK_CONN_LEFT] = src[CHANNEL_BLACK_CONN_RIGHT];
      ch[CHANNEL_BLACK_CONN_RIGHT] = src[CHANNEL_BLACK_CONN_LEFT];
      row[c] = ch;
    }
    out[r] = row;
  }
  return out;
}

/** Legal moves mirrored to (r, 23 - c), in the same order. */
export function mirrorMoves(moves) {
  return moves.map(([r, c]) => [r, BOARD_SIZE - 1 - c]);
}

const f32 = Math.fround;
/** numpy float32 `(a + b) / 2` over float32 inputs. */
const meanF32 = (a, b) => f32(f32(f32(a) + f32(b)) / 2);

export class MirrorAveragingInference {
  /** @param {{evaluate(boardTensorHWC, moves): Promise<{priors: Map, value: number}>}} base */
  constructor(base) {
    this.base = base;
  }

  async evaluate(boardTensorHWC, moves) {
    const mMoves = mirrorMoves(moves);
    const p = await this.base.evaluate(boardTensorHWC, moves);
    const q = await this.base.evaluate(mirrorBoardHWC(boardTensorHWC), mMoves);
    const priors = new Map();
    for (let j = 0; j < moves.length; j++) {
      const key = `${moves[j][0]},${moves[j][1]}`;
      const mKey = `${mMoves[j][0]},${mMoves[j][1]}`;
      priors.set(key, meanF32(p.priors.get(key) ?? 0, q.priors.get(mKey) ?? 0));
    }
    return { priors, value: meanF32(p.value, q.value) };
  }
}
