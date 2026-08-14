#!/usr/bin/env node
/**
 * Node half of the Phase 2 parity measurement.
 *
 *   node tests/parity/run_node_side.mjs <model_dir> <out.json>
 *
 * Produces, for every corpus position: the JavaScript state encoding (as a hash
 * of the exact float32 NCHW buffer the server builds) and the policy/value
 * outputs of Node ONNX Runtime.
 *
 * The model is loaded through `resolveModel` with MODEL_MANIFEST, so this
 * exercises the real validated loading path — hashes, external-data binding and
 * application contract — rather than opening a bare file.
 *
 * It applies no gates and reaches no verdict. Comparison and the pass/fail rule
 * live in compare.mjs, so that neither half of the measurement can decide its
 * own result.
 *
 * Specification: docs/superpowers/2026-08-13-phase2-parity-specification.md
 */
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import * as ort from 'onnxruntime-node';

import {
  TwixtState,
  NUM_CHANNELS,
  BOARD_SIZE,
} from '../../server/gameLogic.js';
import {
  resolveModel,
  assertSessionContract,
} from '../../server/model_manifest.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = join(HERE, '..', '..');

const MAX_MOVES = 576;
const PERSPECTIVE_PROBES = [-1.0, -0.75, -0.25, 0.0, 0.25, 0.75, 1.0];
const EQUIVARIANCE_POSITIONS = 10;
const EQUIVARIANCE_SEED = 20260813;

const sha256 = (buf) => createHash('sha256').update(buf).digest('hex');

/**
 * Build the exact float32 NCHW buffer `AlphaZeroInference.evaluate` feeds.
 *
 * Transcribed from server/inference.js, which documents itself as the only
 * place Node does layout conversion — so this conversion is the thing under
 * test and must not be short-circuited.
 */
function boardNCHW(state) {
  const hwc = state.toTensorHWC();
  const size = BOARD_SIZE;
  const board = new Float32Array(1 * NUM_CHANNELS * size * size);
  for (let c = 0; c < NUM_CHANNELS; c++) {
    for (let r = 0; r < size; r++) {
      for (let col = 0; col < size; col++) {
        board[c * size * size + r * size + col] = hwc[r][col][c];
      }
    }
  }
  return board;
}

function padMoves(moves) {
  const rows = new BigInt64Array(MAX_MOVES);
  const cols = new BigInt64Array(MAX_MOVES);
  const mask = new Float32Array(MAX_MOVES);
  for (let i = 0; i < moves.length && i < MAX_MOVES; i++) {
    rows[i] = BigInt(moves[i][0]);
    cols[i] = BigInt(moves[i][1]);
    mask[i] = 1.0;
  }
  return { rows, cols, mask };
}

/** The server's red-perspective rule, from server/index.js. */
function redPerspective(value, toMove) {
  const red = toMove === 'red' ? value : -value;
  return Math.max(-1, Math.min(1, red));
}

/** Fisher-Yates under mulberry32 — must match the Python half exactly. */
function deterministicPermutation(n, seed) {
  const idx = Array.from({ length: n }, (_, i) => i);
  let a = seed | 0;
  const rand = () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  return idx;
}

async function runOne(session, state, legal) {
  const board = boardNCHW(state);
  const { rows, cols, mask } = padMoves(legal);
  const out = await session.run({
    board: new ort.Tensor('float32', board, [
      1,
      NUM_CHANNELS,
      BOARD_SIZE,
      BOARD_SIZE,
    ]),
    move_rows: new ort.Tensor('int64', rows, [MAX_MOVES]),
    move_cols: new ort.Tensor('int64', cols, [MAX_MOVES]),
    move_mask: new ort.Tensor('float32', mask, [MAX_MOVES]),
  });
  return {
    board,
    logits: Array.from(out.policy_logits.data),
    value: out.value.data[0],
  };
}

async function main() {
  const [modelDirArg, outArg] = process.argv.slice(2);
  if (!modelDirArg || !outArg) {
    console.error('usage: run_node_side.mjs <model_dir> <out.json>');
    process.exit(2);
  }
  const modelDir = resolve(modelDirArg);
  const outPath = resolve(outArg);

  const corpusPath = join(PROJECT_ROOT, 'tests/parity/corpus.json');
  const corpusText = await readFile(corpusPath, 'utf8');
  const corpus = JSON.parse(corpusText);
  const positions = [...corpus.primary, ...corpus.edge];

  // The real loading path, not a bare file open.
  const { manifest, graphPath } = await resolveModel({
    MODEL_MANIFEST: join(modelDir, 'manifest.json'),
  });
  const session = await ort.InferenceSession.create(graphPath);
  assertSessionContract(manifest, session, MAX_MOVES);

  const results = [];
  for (const pos of positions) {
    const state = TwixtState.fromMoves(pos.moves);
    const legal = state.legalMoves();
    const nLegal = legal.length;
    const { board, logits, value } = await runOne(session, state, legal);

    let maskTailOk = true;
    for (let i = nLegal; i < MAX_MOVES; i++) {
      if (logits[i] !== -1e9) {
        maskTailOk = false;
        break;
      }
    }

    results.push({
      id: pos.id,
      stratum: pos.stratum,
      ply: pos.ply,
      to_move: state.toMove,
      n_legal: nLegal,
      legal_moves: legal.map((m) => [m[0], m[1]]),
      encoding_sha256: sha256(
        Buffer.from(board.buffer, board.byteOffset, board.byteLength)
      ),
      encoding_shape: [1, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE],
      ort_node: {
        logits: logits.slice(0, nLegal),
        value,
        mask_tail_all_neg1e9: maskTailOk,
        mask_tail_count: MAX_MOVES - nLegal,
      },
      red_perspective_probes: PERSPECTIVE_PROBES.map((v) =>
        redPerspective(v, state.toMove)
      ),
    });
  }

  const equivariance = [];
  for (const pos of corpus.primary.slice(0, EQUIVARIANCE_POSITIONS)) {
    const state = TwixtState.fromMoves(pos.moves);
    const legal = state.legalMoves();
    const perm = deterministicPermutation(legal.length, EQUIVARIANCE_SEED);
    const permuted = perm.map((i) => legal[i]);
    const { logits } = await runOne(session, state, permuted);
    equivariance.push({
      id: pos.id,
      permutation: perm,
      permuted_logits: logits.slice(0, legal.length),
    });
  }

  const payload = {
    schema: 'twixt-parity-side/1',
    side: 'node',
    specification: 'docs/superpowers/2026-08-13-phase2-parity-specification.md',
    corpus_sha256: sha256(Buffer.from(corpusText, 'utf8')),
    model_dir: modelDir,
    model_id: manifest.model_id,
    graph_sha256: manifest.graph.sha256,
    scratch_export_path: process.env.PHASE2_SCRATCH_EXPORT || '(not recorded)',
    source_checkpoint_sha1: manifest.provenance.source_checkpoint_sha1,
    loaded_through_manifest: true,
    environment: {
      node: process.version,
      platform: `${process.platform} ${process.arch}`,
      onnxruntime_node: JSON.parse(
        await readFile(
          join(PROJECT_ROOT, 'node_modules/onnxruntime-node/package.json'),
          'utf8'
        )
      ).version,
    },
    constants: {
      num_channels: NUM_CHANNELS,
      board_size: BOARD_SIZE,
      max_moves: MAX_MOVES,
      perspective_probes: PERSPECTIVE_PROBES,
      equivariance_positions: EQUIVARIANCE_POSITIONS,
      equivariance_seed: EQUIVARIANCE_SEED,
    },
    positions: results,
    equivariance,
  };

  await mkdir(dirname(outPath), { recursive: true });
  await writeFile(outPath, JSON.stringify(payload));
  console.log(`wrote ${outPath} (${results.length} positions)`);
}

await main();
