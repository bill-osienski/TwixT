#!/usr/bin/env node
/**
 * V3, product side: MirrorAveragingInference over the pinned hard model, over the Phase 2 corpus.
 *
 *   node tests/parity/htb_mirror/node_side.mjs <out.json>
 *
 * The model is resolved through the hard model's own validated path (resolveHardModel, the pinned
 * HARD_MODEL_ID) and its session contract asserted, as server/index.js does at startup. No gates here.
 */
import { readFile, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { TwixtState } from '../../../server/gameLogic.js';
import { AlphaZeroInference } from '../../../server/inference.js';
import { MirrorAveragingInference } from '../../../server/mirror_inference.js';
import { assertSessionContract, resolveModel } from '../../../server/model_manifest.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..', '..');
const out = process.argv[2];
// H-OP COPY (tests/htb_onpolicy_v3_spec.py): the challenger's exported pair is resolved from its directory
// (argv 3) through resolveModel; otherwise tests/parity/htb_mirror/node_side.mjs.
const hard = await resolveModel({ MODEL_MANIFEST: join(process.argv[3], 'manifest.json') });
const base = new AlphaZeroInference(hard.graphPath);
await base.load();
assertSessionContract(hard.manifest, base.session, base.maxMoves);
const mirror = new MirrorAveragingInference(base);
const corpus = JSON.parse(await readFile(join(ROOT, 'tests/parity/corpus.json'), 'utf8'));
const positions = [];
for (const p of [...corpus.primary, ...corpus.edge]) {
  const state = TwixtState.fromMoves(p.moves);
  const legal = state.legalMoves();
  const { priors, value } = await mirror.evaluate(state.toTensorHWC(), legal);
  positions.push({
    id: p.id,
    to_move: state.toMove,
    n_legal: legal.length,
    legal_moves: legal,
    priors: legal.map((m) => priors.get(`${m[0]},${m[1]}`)),
    value,
  });
}
await writeFile(out, JSON.stringify({ side: 'product MirrorAveragingInference(H-OP challenger)', model_id: hard.manifest.model_id, node: process.version, positions }));
console.log(`wrote ${out} (${positions.length} positions)`);
