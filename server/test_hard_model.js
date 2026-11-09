/**
 * The hard model's own selection path (integration proposal §4.7) and the
 * retained rollback pins (§8). Manifest resolution and file hashing only: no
 * ONNX session is created here.
 */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { after, before, describe, it } from 'node:test';
import { fileURLToPath } from 'node:url';

import {
  DEFAULT_MANIFEST_PATH,
  DEFAULT_MODEL_ID,
  HARD_MODEL_ID,
  ModelManifestError,
  computeModelId,
  loadManifest,
  resolveHardManifestPath,
  resolveHardModel,
  resolveManifestPath,
  resolveModel,
  validateArtifact,
} from './model_manifest.js';

const REPO_ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const sha256 = (buf) => createHash('sha256').update(buf).digest('hex');

// A minimal synthetic ONNX graph: one initializer whose external data names the
// sidecar. Enough for validateArtifact's structural binding; no session.
const pbVarint = (n) => {
  const out = [];
  let v = n;
  do {
    let b = v % 128;
    v = Math.floor(v / 128);
    if (v > 0) b |= 0x80;
    out.push(b);
  } while (v > 0);
  return Buffer.from(out);
};
const pbLenDelim = (field, payload) => Buffer.concat([pbVarint(field * 8 + 2), pbVarint(payload.length), payload]);
const pbString = (field, s) => pbLenDelim(field, Buffer.from(s, 'utf8'));
const pbEntry = (k, v) => Buffer.concat([pbString(1, k), pbString(2, v)]);
const syntheticGraph = (tag) =>
  pbLenDelim(
    7,
    Buffer.concat([
      pbString(2, `graph-${tag}`),
      pbLenDelim(
        5,
        Buffer.concat([pbString(8, 'w0'), pbLenDelim(13, pbEntry('location', 'model.onnx.data')), pbVarint(14 * 8), pbVarint(1)])
      ),
    ])
  );

async function makeFixture(dir, tag) {
  await mkdir(dir, { recursive: true });
  const graph = syntheticGraph(tag);
  const data = Buffer.from(`FAKE-WEIGHTS-${tag}`);
  await writeFile(join(dir, 'model.onnx'), graph);
  await writeFile(join(dir, 'model.onnx.data'), data);
  const manifest = {
    manifest_version: 1,
    model_id: computeModelId(sha256(graph), sha256(data)),
    description: `test fixture ${tag}`,
    graph: { filename: 'model.onnx', size_bytes: graph.length, sha256: sha256(graph) },
    external_data: { filename: 'model.onnx.data', size_bytes: data.length, sha256: sha256(data) },
    provenance: {
      source_checkpoint_path: 'unknown',
      source_checkpoint_sha1: 'unknown',
      export_commit: 'unknown',
      exporter_config: 'unknown',
      runtime_versions: 'unknown',
    },
    contract: {
      inputs: [
        { name: 'board', type: 'float32', shape: [1, 30, 24, 24] },
        { name: 'move_rows', type: 'int64', shape: [576] },
        { name: 'move_cols', type: 'int64', shape: [576] },
        { name: 'move_mask', type: 'float32', shape: [576] },
      ],
      outputs: [
        { name: 'policy_logits', type: 'float32', shape: [576] },
        { name: 'value', type: 'float32', shape: [] },
      ],
      board_shape: [1, 30, 24, 24],
      max_moves: 576,
      onnxruntime_node: '^1.20.0',
    },
    claims: 'Identity only.',
  };
  const path = join(dir, 'manifest.json');
  await writeFile(path, JSON.stringify(manifest));
  return { path, id: manifest.model_id };
}

describe('the hard model has its own selection path', () => {
  let dir;
  let A;
  let B;
  before(async () => {
    dir = await mkdtemp(join(tmpdir(), 'twixt-hard-'));
    A = await makeFixture(join(dir, 'a'), 'a');
    B = await makeFixture(join(dir, 'b'), 'b');
  });
  after(async () => rm(dir, { recursive: true, force: true }));

  it('nothing is pinned until the export (G2); an unpinned hard path refuses with a distinct code', () => {
    assert.equal(HARD_MODEL_ID, null);
    assert.throws(
      () => resolveHardManifestPath({}),
      (e) => e instanceof ModelManifestError && e.code === 'HARD_MODEL_UNPINNED' && /HARD_MODE=legacy/.test(e.message)
    );
  });

  it('a pinned id resolves to models/<id>/manifest.json, application-relative', () => {
    assert.equal(resolveHardManifestPath({}, '/elsewhere', 'abcdef0123456789'), join(REPO_ROOT, 'models', 'abcdef0123456789', 'manifest.json'));
  });

  it('HARD_MODEL_MANIFEST overrides the hard path, relative to cwd when relative', () => {
    assert.equal(resolveHardManifestPath({ HARD_MODEL_MANIFEST: '/x/manifest.json' }), '/x/manifest.json');
    assert.equal(resolveHardManifestPath({ HARD_MODEL_MANIFEST: 'rel/m.json' }, '/cwd'), '/cwd/rel/m.json');
  });

  it('MODEL_MANIFEST never changes the hard model', async () => {
    assert.throws(() => resolveHardManifestPath({ MODEL_MANIFEST: A.path }), (e) => e.code === 'HARD_MODEL_UNPINNED');
    const hard = await resolveHardModel({ MODEL_MANIFEST: A.path, HARD_MODEL_MANIFEST: B.path });
    assert.equal(hard.manifest.model_id, B.id);
  });

  it('HARD_MODEL_MANIFEST never changes the default model', async () => {
    assert.equal(resolveManifestPath({ HARD_MODEL_MANIFEST: B.path }), DEFAULT_MANIFEST_PATH);
    const def = await resolveModel({ MODEL_MANIFEST: A.path, HARD_MODEL_MANIFEST: B.path });
    assert.equal(def.manifest.model_id, A.id);
  });

  it('the hard path gets the identical validation (a tampered sidecar is refused)', async () => {
    const t = await makeFixture(join(dir, 't'), 't');
    await writeFile(join(dir, 't', 'model.onnx.data'), Buffer.from('FAKE-WEIGHTS-T'));
    await assert.doesNotReject(resolveHardModel({ HARD_MODEL_MANIFEST: A.path })); // control
    await assert.rejects(
      resolveHardModel({ HARD_MODEL_MANIFEST: t.path }),
      (e) => e instanceof ModelManifestError && e.code === 'DATA_HASH_MISMATCH'
    );
    assert.notEqual(A.id, B.id);
  });
});

describe('the retained rollback pins are byte-identical (§8)', () => {
  for (const id of [DEFAULT_MODEL_ID, '1d64027db521a50f']) {
    it(`models/${id}: the manifest validates and both files hash to its pair`, async () => {
      const manifestPath = join(REPO_ROOT, 'models', id, 'manifest.json');
      const manifest = await loadManifest(manifestPath);
      assert.equal(manifest.model_id, id);
      await validateArtifact(manifest, dirname(manifestPath)); // sizes and SHA-256 of graph and sidecar
    });
  }
  it('the served default is still c34b7ff3297c785a', () => {
    assert.equal(DEFAULT_MODEL_ID, 'c34b7ff3297c785a');
  });
});
