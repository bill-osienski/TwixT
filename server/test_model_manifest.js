/**
 * Tests for the model manifest loading path.
 *
 * Run with: node --test server/test_model_manifest.js
 *
 * The point of this module is that it FAILS. Most of these are negative tests:
 * each one breaks exactly one property of a valid artifact and asserts the
 * loader refuses to serve it, with a distinguishable code. A loader that
 * degrades gracefully here is the defect, not the feature.
 *
 * Fixtures use the REAL pinned graph wherever possible (82 KB, cheap to copy)
 * so the protobuf walk is exercised against genuine ONNX bytes, and synthetic
 * hand-encoded graphs where a specific malformation has to be constructed.
 */
import { describe, it, before, after } from 'node:test';
import assert from 'node:assert';
import { mkdtemp, rm, writeFile, readFile, mkdir, copyFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, dirname, isAbsolute } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { spawn } from 'node:child_process';

import {
  DEFAULT_MANIFEST_PATH,
  DEFAULT_MODEL_ID,
  ModelManifestError,
  applicationContract,
  computeModelId,
  externalDataLocations,
  resolveManifestPath,
  loadManifest,
  validateArtifact,
  assertSessionContract,
  resolveModel,
} from './model_manifest.js';
import { NUM_CHANNELS, BOARD_SIZE } from './gameLogic.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = join(HERE, '..');
const REAL_GRAPH = join(REPO_ROOT, 'models', DEFAULT_MODEL_ID, 'model.onnx');

const sha256 = (buf) => createHash('sha256').update(buf).digest('hex');

// --- minimal protobuf encoders, for constructing malformed graphs -----------

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
const pbLenDelim = (field, payload) =>
  Buffer.concat([pbVarint(field * 8 + 2), pbVarint(payload.length), payload]);
const pbString = (field, s) => pbLenDelim(field, Buffer.from(s, 'utf8'));
const pbVarintField = (field, n) => Buffer.concat([pbVarint(field * 8), pbVarint(n)]);
const pbEntry = (k, v) => Buffer.concat([pbString(1, k), pbString(2, v)]);

/** A TensorProto initializer (GraphProto field 5) loading from `location`. */
const pbInitializer = (name, location) =>
  pbLenDelim(
    5,
    Buffer.concat([
      pbString(8, name), // TensorProto.name
      pbLenDelim(13, pbEntry('location', location)), // TensorProto.external_data
      pbVarintField(14, 1), // TensorProto.data_location = EXTERNAL
    ])
  );

/**
 * Build a ModelProto.
 *
 * `docString` writes ModelProto.doc_string (field 6) — the decoy vector: a
 * filename that appears in the serialized bytes without any initializer
 * loading from it.
 *
 * `strayLocationKey` encodes an external-data entry under an unknown top-level
 * field, i.e. somewhere the structured walk does not reach.
 */
function buildOnnx({ locations = ['model.onnx.data'], docString = null, strayLocationKey = false }) {
  const graph = Buffer.concat([
    pbString(2, 'test-graph'),
    ...locations.map((loc, i) => pbInitializer(`w${i}`, loc)),
  ]);
  const parts = [pbLenDelim(7, graph)];
  if (docString !== null) parts.push(pbString(6, docString));
  if (strayLocationKey) parts.push(pbLenDelim(99, pbEntry('location', 'elsewhere.bin')));
  return Buffer.concat(parts);
}

// --- fixtures ---------------------------------------------------------------

const VALID_CONTRACT = {
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
};

/** A session stand-in exposing exactly what assertSessionContract reads. */
const fakeSession = (inputs, outputs) => ({ inputMetadata: inputs, outputMetadata: outputs });

/**
 * Write a self-consistent artifact directory and its manifest.
 *
 * By default the graph is a copy of the real pinned graph, so the protobuf walk
 * runs against genuine bytes. `graphBytes` substitutes a synthetic graph.
 * `mutate` then breaks exactly one thing.
 */
async function makeFixture(dir, { graphBytes = null, mutate = () => {} } = {}) {
  const graphName = 'model.onnx';
  const dataName = 'model.onnx.data';

  const graph = graphBytes ?? (await readFile(REAL_GRAPH));
  const data = Buffer.from('FAKE-WEIGHTS-0123456789');

  await writeFile(join(dir, graphName), graph);
  await writeFile(join(dir, dataName), data);

  const graphHash = sha256(graph);
  const dataHash = sha256(data);

  const manifest = {
    manifest_version: 1,
    model_id: computeModelId(graphHash, dataHash),
    description: 'test fixture',
    graph: { filename: graphName, size_bytes: graph.length, sha256: graphHash },
    external_data: { filename: dataName, size_bytes: data.length, sha256: dataHash },
    provenance: {
      source_checkpoint_path: 'unknown',
      source_checkpoint_sha1: 'unknown',
      export_commit: 'unknown',
      exporter_config: 'unknown',
      runtime_versions: 'unknown',
    },
    contract: structuredClone(VALID_CONTRACT),
    claims: 'Identity only.',
  };

  await mutate(manifest, { dir, graphName, dataName, graph, data });

  const manifestPath = join(dir, 'manifest.json');
  await writeFile(manifestPath, JSON.stringify(manifest, null, 2));
  return manifestPath;
}

/** Assert the call throws a ModelManifestError carrying exactly `code`. */
async function assertRejectsWithCode(fn, code) {
  try {
    await fn();
  } catch (err) {
    assert.ok(
      err instanceof ModelManifestError,
      `expected ModelManifestError, got ${err?.name}: ${err?.message}`
    );
    assert.strictEqual(err.code, code, `expected code ${code}, got ${err.code}: ${err.message}`);
    return err;
  }
  assert.fail(`expected ${code}, but the call resolved`);
}

/** Load + validate a fixture manifest in one step, the way the server does. */
async function loadAndValidate(manifestPath) {
  const manifest = await loadManifest(manifestPath);
  return validateArtifact(manifest, dirname(manifestPath));
}

describe('manifest path resolution', () => {
  it('defaults to an application-relative path, never cwd', () => {
    assert.ok(isAbsolute(DEFAULT_MANIFEST_PATH), 'default must be absolute');
    const fromRepoRoot = resolveManifestPath({}, REPO_ROOT);
    const fromElsewhere = resolveManifestPath({}, tmpdir());
    assert.strictEqual(fromRepoRoot, fromElsewhere);
    assert.strictEqual(fromRepoRoot, DEFAULT_MANIFEST_PATH);
  });

  it('honours MODEL_MANIFEST as an override', () => {
    const p = resolveManifestPath({ MODEL_MANIFEST: '/tmp/other/manifest.json' }, REPO_ROOT);
    assert.strictEqual(p, '/tmp/other/manifest.json');
  });

  it('resolves a relative MODEL_MANIFEST against cwd, not the app', () => {
    const p = resolveManifestPath({ MODEL_MANIFEST: 'rel/manifest.json' }, '/some/cwd');
    assert.strictEqual(p, '/some/cwd/rel/manifest.json');
  });

  it('ignores MODEL_PATH entirely — it is no longer a serving override', () => {
    const p = resolveManifestPath({ MODEL_PATH: '/tmp/sneaky.onnx' }, REPO_ROOT);
    assert.strictEqual(p, DEFAULT_MANIFEST_PATH);
  });
});

describe('model id addresses the whole pair', () => {
  it('is derived from both hashes, not the graph alone', () => {
    const g = 'a'.repeat(64);
    const d1 = 'b'.repeat(64);
    const d2 = 'c'.repeat(64);
    // The collision the graph-only scheme allowed: identical topology and
    // external-data filenames produce identical graph bytes for any weights.
    assert.notStrictEqual(computeModelId(g, d1), computeModelId(g, d2));
    assert.strictEqual(computeModelId(g, d1), computeModelId(g, d1));
    assert.match(computeModelId(g, d1), /^[0-9a-f]{16}$/);
  });

  it('the committed baseline id follows from its own hashes', async () => {
    const { manifest } = await resolveModel({}, REPO_ROOT);
    assert.strictEqual(
      manifest.model_id,
      computeModelId(manifest.graph.sha256, manifest.external_data.sha256)
    );
    assert.strictEqual(manifest.model_id, DEFAULT_MODEL_ID);
  });
});

describe('committed baseline artifact', () => {
  it('the default manifest exists, validates, and its pair hashes match', async () => {
    const { manifest, graphPath, dataPath } = await resolveModel({}, REPO_ROOT);

    assert.ok(existsSync(graphPath), `graph missing: ${graphPath}`);
    assert.ok(existsSync(dataPath), `sidecar missing: ${dataPath}`);

    // Re-derive independently of the loader so a bug in the loader cannot
    // make this test agree with it.
    assert.strictEqual(sha256(await readFile(graphPath)), manifest.graph.sha256);
    assert.strictEqual(sha256(await readFile(dataPath)), manifest.external_data.sha256);
  });

  it('every external-data reference in the real graph names the declared sidecar', async () => {
    const { manifest, graphPath } = await resolveModel({}, REPO_ROOT);
    const locations = externalDataLocations(await readFile(graphPath));
    assert.strictEqual(locations.length, 33);
    assert.deepStrictEqual([...new Set(locations)], [manifest.external_data.filename]);
  });

  it('records provenance as unknown rather than guessing it', async () => {
    const { manifest } = await resolveModel({}, REPO_ROOT);
    for (const key of Object.keys(manifest.provenance)) {
      assert.strictEqual(
        manifest.provenance[key],
        'unknown',
        `${key} must be "unknown" for the baseline, not reconstructed`
      );
    }
  });

  it('attaches no parity or strength claim', async () => {
    const { manifest } = await resolveModel({}, REPO_ROOT);
    assert.match(manifest.claims, /identity/i);
    assert.match(manifest.claims, /no parity/i);
  });

  it('the live session matches the declared contract exactly', async () => {
    // The relocation risk: ONNX resolves external data relative to the graph
    // file's directory. If the sidecar did not move with the graph under its
    // exact basename, this is where it shows up.
    const ort = await import('onnxruntime-node');
    const { graphPath, manifest } = await resolveModel({}, REPO_ROOT);
    const session = await ort.InferenceSession.create(graphPath);
    assert.doesNotThrow(() => assertSessionContract(manifest, session, 576));
  });
});

describe('fail-loud validation', () => {
  let dir;
  before(async () => {
    dir = await mkdtemp(join(tmpdir(), 'twixt-manifest-'));
  });
  after(async () => {
    await rm(dir, { recursive: true, force: true });
  });

  const fixtureDir = async (name) => {
    const d = join(dir, name);
    await mkdir(d, { recursive: true });
    return d;
  };

  it('accepts a well-formed fixture (control)', async () => {
    const d = await fixtureDir('ok');
    const manifestPath = await makeFixture(d);
    const { graphPath, dataPath } = await loadAndValidate(manifestPath);
    assert.strictEqual(graphPath, join(d, 'model.onnx'));
    assert.strictEqual(dataPath, join(d, 'model.onnx.data'));
  });

  it('MANIFEST_MISSING when the manifest does not exist', async () => {
    await assertRejectsWithCode(
      () => loadManifest(join(dir, 'nope', 'manifest.json')),
      'MANIFEST_MISSING'
    );
  });

  it('MANIFEST_UNREADABLE when the manifest is not JSON', async () => {
    const d = await fixtureDir('badjson');
    const p = join(d, 'manifest.json');
    await writeFile(p, '{ not json');
    await assertRejectsWithCode(() => loadManifest(p), 'MANIFEST_UNREADABLE');
  });

  it('MANIFEST_INVALID when a required field is absent', async () => {
    const d = await fixtureDir('missingfield');
    const p = await makeFixture(d, { mutate: (m) => delete m.contract.max_moves });
    await assertRejectsWithCode(() => loadManifest(p), 'MANIFEST_INVALID');
  });

  it('MANIFEST_INVALID on an unsupported manifest_version', async () => {
    const d = await fixtureDir('badversion');
    const p = await makeFixture(d, { mutate: (m) => (m.manifest_version = 99) });
    await assertRejectsWithCode(() => loadManifest(p), 'MANIFEST_INVALID');
  });

  it('MANIFEST_INVALID when a filename is a path rather than a basename', async () => {
    const d = await fixtureDir('pathescape');
    const p = await makeFixture(d, {
      mutate: (m) => {
        m.external_data.filename = '../outside/model.onnx.data';
        m.model_id = computeModelId(m.graph.sha256, m.external_data.sha256);
      },
    });
    await assertRejectsWithCode(() => loadManifest(p), 'MANIFEST_INVALID');
  });

  it('MODEL_ID_MISMATCH when the id does not follow from the declared hashes', async () => {
    const d = await fixtureDir('badid');
    const p = await makeFixture(d, { mutate: (m) => (m.model_id = 'deadbeefdeadbeef') });
    await assertRejectsWithCode(() => loadManifest(p), 'MODEL_ID_MISMATCH');
  });

  it('GRAPH_MISSING when the graph file is absent', async () => {
    const d = await fixtureDir('nograph');
    const p = await makeFixture(d);
    await rm(join(d, 'model.onnx'));
    await assertRejectsWithCode(() => loadAndValidate(p), 'GRAPH_MISSING');
  });

  it('GRAPH_SIZE_MISMATCH when the declared size is wrong', async () => {
    const d = await fixtureDir('graphsize');
    const p = await makeFixture(d, { mutate: (m) => (m.graph.size_bytes += 1) });
    await assertRejectsWithCode(() => loadAndValidate(p), 'GRAPH_SIZE_MISMATCH');
  });

  it('GRAPH_HASH_MISMATCH when the graph bytes change', async () => {
    const d = await fixtureDir('graphhash');
    const p = await makeFixture(d);
    // Same length, different bytes — size alone would not catch this.
    const graphPath = join(d, 'model.onnx');
    const bytes = await readFile(graphPath);
    bytes[bytes.length - 1] ^= 0xff;
    await writeFile(graphPath, bytes);
    await assertRejectsWithCode(() => loadAndValidate(p), 'GRAPH_HASH_MISMATCH');
  });

  it('DATA_MISSING when the sidecar is absent', async () => {
    const d = await fixtureDir('nodata');
    const p = await makeFixture(d);
    await rm(join(d, 'model.onnx.data'));
    await assertRejectsWithCode(() => loadAndValidate(p), 'DATA_MISSING');
  });

  it('DATA_SIZE_MISMATCH when the declared sidecar size is wrong', async () => {
    const d = await fixtureDir('datasize');
    const p = await makeFixture(d, { mutate: (m) => (m.external_data.size_bytes += 1) });
    await assertRejectsWithCode(() => loadAndValidate(p), 'DATA_SIZE_MISMATCH');
  });

  it('DATA_HASH_MISMATCH when the weights change', async () => {
    const d = await fixtureDir('datahash');
    const p = await makeFixture(d);
    const dataPath = join(d, 'model.onnx.data');
    const bytes = await readFile(dataPath);
    bytes[0] ^= 0xff;
    await writeFile(dataPath, bytes);
    await assertRejectsWithCode(() => loadAndValidate(p), 'DATA_HASH_MISMATCH');
  });
});

describe('external-data binding is structural, not textual', () => {
  let dir;
  before(async () => {
    dir = await mkdtemp(join(tmpdir(), 'twixt-extdata-'));
  });
  after(async () => {
    await rm(dir, { recursive: true, force: true });
  });

  const fixtureDir = async (name) => {
    const d = join(dir, name);
    await mkdir(d, { recursive: true });
    return d;
  };

  it('parses locations out of a synthetic graph', () => {
    const locs = externalDataLocations(buildOnnx({ locations: ['a.bin', 'b.bin'] }));
    assert.deepStrictEqual(locs, ['a.bin', 'b.bin']);
  });

  it('rejects a decoy filename carried in the doc string', async () => {
    // The attack a substring search cannot see: the declared sidecar name
    // appears in the serialized bytes, but every initializer loads something
    // else. Both files hash correctly, so only a structural check catches it.
    const graphBytes = buildOnnx({
      locations: ['weights.bin'],
      docString: 'model.onnx.data',
    });
    assert.ok(
      graphBytes.includes(Buffer.from('model.onnx.data')),
      'fixture must contain the decoy string, or it is not testing the attack'
    );

    const d = await fixtureDir('decoy');
    const p = await makeFixture(d, { graphBytes });
    await assertRejectsWithCode(() => loadAndValidate(p), 'EXTERNAL_REF_MISMATCH');
  });

  it('rejects a graph loading from more than one sidecar', async () => {
    const d = await fixtureDir('twolocations');
    const p = await makeFixture(d, {
      graphBytes: buildOnnx({ locations: ['model.onnx.data', 'extra.bin'] }),
    });
    await assertRejectsWithCode(() => loadAndValidate(p), 'EXTERNAL_REF_MISMATCH');
  });

  it('rejects a graph that references no external data at all', async () => {
    const d = await fixtureDir('noexternal');
    const p = await makeFixture(d, { graphBytes: buildOnnx({ locations: [] }) });
    await assertRejectsWithCode(() => loadAndValidate(p), 'EXTERNAL_REF_MISMATCH');
  });

  it('rejects external-data entries hidden outside graph.initializer', async () => {
    const d = await fixtureDir('stray');
    const p = await makeFixture(d, {
      graphBytes: buildOnnx({ locations: ['model.onnx.data'], strayLocationKey: true }),
    });
    await assertRejectsWithCode(() => loadAndValidate(p), 'EXTERNAL_REF_MISMATCH');
  });

  it('GRAPH_UNPARSEABLE on bytes that are not a protobuf message', async () => {
    const d = await fixtureDir('garbage');
    const p = await makeFixture(d, { graphBytes: Buffer.from('not a protobuf at all, sorry') });
    await assertRejectsWithCode(() => loadAndValidate(p), 'GRAPH_UNPARSEABLE');
  });
});

describe('tensor contract is enforced, not merely recorded', () => {
  const live = fakeSession(VALID_CONTRACT.inputs, VALID_CONTRACT.outputs);
  const manifestWith = (contract) => ({ contract });

  const expectMismatch = (manifest, session = live, wrapperMaxMoves = 576) =>
    assert.throws(
      () => assertSessionContract(manifest, session, wrapperMaxMoves),
      (err) => err instanceof ModelManifestError && err.code === 'CONTRACT_MISMATCH'
    );

  it('accepts the matching contract in any tensor order', () => {
    const shuffled = fakeSession(
      [...VALID_CONTRACT.inputs].reverse(),
      [...VALID_CONTRACT.outputs].reverse()
    );
    assert.doesNotThrow(() =>
      assertSessionContract(manifestWith(VALID_CONTRACT), shuffled, 576)
    );
  });

  it('rejects a missing input', () => {
    const c = structuredClone(VALID_CONTRACT);
    c.inputs = c.inputs.slice(0, 3);
    expectMismatch(manifestWith(c));
  });

  it('rejects a wrong tensor type', () => {
    const c = structuredClone(VALID_CONTRACT);
    c.inputs.find((t) => t.name === 'move_rows').type = 'int32';
    expectMismatch(manifestWith(c));
  });

  it('rejects a 24-channel board against the real 30-channel session', () => {
    // The reported defect: names alone matched, so this was accepted.
    const c = structuredClone(VALID_CONTRACT);
    c.inputs.find((t) => t.name === 'board').shape = [1, 24, 24, 24];
    c.board_shape = [1, 24, 24, 24];
    expectMismatch(manifestWith(c));
  });

  it('rejects a 512-move policy against the real 576-move session', () => {
    const c = structuredClone(VALID_CONTRACT);
    for (const t of [...c.inputs, ...c.outputs]) {
      if (t.name !== 'board' && t.name !== 'value') t.shape = [512];
    }
    c.max_moves = 512;
    expectMismatch(manifestWith(c));
  });

  it('rejects board_shape that contradicts its own declared board tensor', () => {
    const c = structuredClone(VALID_CONTRACT);
    c.board_shape = [1, 24, 24, 24];
    expectMismatch(manifestWith(c));
  });

  it('rejects max_moves that contradicts its own declared move tensors', () => {
    const c = structuredClone(VALID_CONTRACT);
    c.max_moves = 512;
    expectMismatch(manifestWith(c));
  });

  it('rejects an artifact whose move count disagrees with AlphaZeroInference', () => {
    // maxMoves sizes the move buffers and caps the policy read; a mismatch
    // either over-reads the output buffer or silently drops legal moves.
    // The artifact is internally fine here — this server just cannot feed it.
    assert.throws(
      () => assertSessionContract(manifestWith(VALID_CONTRACT), live, 512),
      (err) => err instanceof ModelManifestError && err.code === 'APPLICATION_MISMATCH'
    );
  });
});

describe('application compatibility, not just self-consistency', () => {
  // In every case below the manifest and the session AGREE with each other.
  // Steps 1 and 2 of the check pass. They are rejected because this server's
  // own code cannot consume them: AlphaZeroInference feeds a board built from
  // NUM_CHANNELS and BOARD_SIZE, supplies four named tensors, and reads
  // results.policy_logits and results.value.
  const colluding = (mutate) => {
    const contract = structuredClone(VALID_CONTRACT);
    mutate(contract);
    return {
      manifest: { contract },
      session: fakeSession(contract.inputs, contract.outputs),
    };
  };

  const expectApplicationMismatch = (mutate, maxMoves = 576) => {
    const { manifest, session } = colluding(mutate);
    assert.throws(
      () => assertSessionContract(manifest, session, maxMoves),
      (err) => err instanceof ModelManifestError && err.code === 'APPLICATION_MISMATCH'
    );
  };

  it('the derived application contract matches the pinned baseline', async () => {
    const { manifest } = await resolveModel({}, REPO_ROOT);
    const required = applicationContract(576);
    assert.deepStrictEqual(manifest.contract.inputs, required.inputs);
    assert.deepStrictEqual(manifest.contract.outputs, required.outputs);
    // Guards against the contract being written to match the artifact rather
    // than the application: these come from gameLogic, not from the manifest.
    assert.deepStrictEqual(required.inputs[0].shape, [1, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE]);
    // The engine supports other sizes (curriculum training runs 8..24); the
    // served product is pinned to the official 24x24, and the contract follows
    // gameLogic rather than anything the artifact declares.
    assert.strictEqual(BOARD_SIZE, 24, 'the served board is the official 24x24');
  });

  it('rejects a 24-channel model whose manifest agrees with it', () => {
    // The reported defect. Both sides say 24 channels; AlphaZeroInference
    // still builds a 30-channel board and the run would fail at inference.
    expectApplicationMismatch((c) => {
      c.inputs.find((t) => t.name === 'board').shape = [1, 24, BOARD_SIZE, BOARD_SIZE];
      c.board_shape = [1, 24, BOARD_SIZE, BOARD_SIZE];
    });
  });

  it('rejects an output named `score` when the server reads `value`', () => {
    expectApplicationMismatch((c) => {
      c.outputs.find((t) => t.name === 'value').name = 'score';
    });
  });

  it('rejects a renamed input — caught earlier, as an internal inconsistency', () => {
    // Rejected, but by step 2 rather than step 3: max_moves is defined by
    // reference to move_mask, so renaming it makes the manifest incoherent
    // before compatibility is reached. Recorded so the earlier code is a
    // documented interaction rather than a surprise.
    const { manifest, session } = colluding((c) => {
      c.inputs.find((t) => t.name === 'move_mask').name = 'mask';
    });
    assert.throws(
      () => assertSessionContract(manifest, session, 576),
      (err) => err instanceof ModelManifestError && err.code === 'CONTRACT_MISMATCH'
    );
  });

  it('rejects int32 move indices when the server supplies int64', () => {
    expectApplicationMismatch((c) => {
      c.inputs.find((t) => t.name === 'move_rows').type = 'int32';
    });
  });

  it('rejects a board size other than the served 24', () => {
    expectApplicationMismatch((c) => {
      c.inputs.find((t) => t.name === 'board').shape = [1, NUM_CHANNELS, 19, 19];
      c.board_shape = [1, NUM_CHANNELS, 19, 19];
    });
  });

  it('rejects a self-consistent 512-move artifact against a 576-move server', () => {
    expectApplicationMismatch((c) => {
      for (const t of [...c.inputs, ...c.outputs]) {
        if (t.name !== 'board' && t.name !== 'value') t.shape = [512];
      }
      c.max_moves = 512;
    });
  });

  it('rejects an extra output the server does not handle', () => {
    expectApplicationMismatch((c) => {
      c.outputs.push({ name: 'aux_head', type: 'float32', shape: [8] });
    });
  });

  it('rejects a non-scalar value head', () => {
    expectApplicationMismatch((c) => {
      c.outputs.find((t) => t.name === 'value').shape = [1];
    });
  });

  it('requires the application move count to be supplied at all', () => {
    const { manifest, session } = colluding(() => {});
    assert.throws(
      () => assertSessionContract(manifest, session, undefined),
      (err) => err instanceof ModelManifestError && err.code === 'APPLICATION_MISMATCH'
    );
  });
});

// ---------------------------------------------------------------------------
// Entry-point regression tests.
//
// The properties below were verified by hand once. Source-text assertions
// cannot keep them true: they check that the old code is gone, not that the
// new behaviour works. These start the actual processes.
// ---------------------------------------------------------------------------

const cleanEnv = () => {
  const env = { ...process.env };
  delete env.MODEL_MANIFEST;
  delete env.MODEL_PATH;
  return env;
};

/**
 * Run a script to completion, or until `until` matches its output, then signal
 * it. Long-lived servers never exit on their own, so `until` is how a test
 * observes startup and then stops it.
 */
function runNode(script, { env = {}, cwd = REPO_ROOT, until = null, signal = 'SIGTERM' } = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [script], { cwd, env: { ...cleanEnv(), ...env } });
    let stdout = '';
    let stderr = '';
    let settled = false;

    const timer = setTimeout(() => {
      child.kill('SIGKILL');
      if (!settled) {
        settled = true;
        reject(new Error(`timed out; stdout=${stdout}\nstderr=${stderr}`));
      }
    }, 30000);
    timer.unref();

    const maybeStop = () => {
      if (until && (until.test(stdout) || until.test(stderr))) child.kill(signal);
    };
    child.stdout.on('data', (d) => {
      stdout += d;
      maybeStop();
    });
    child.stderr.on('data', (d) => {
      stderr += d;
      maybeStop();
    });
    child.on('exit', (code) => {
      clearTimeout(timer);
      if (!settled) {
        settled = true;
        resolve({ code, stdout, stderr });
      }
    });
  });
}

describe('inference server entry point', () => {
  let dir;
  let tamperedManifest;

  before(async () => {
    dir = await mkdtemp(join(tmpdir(), 'twixt-entry-'));
    const src = join(REPO_ROOT, 'models', DEFAULT_MODEL_ID);
    await mkdir(join(dir, 'tampered'), { recursive: true });
    for (const f of ['manifest.json', 'model.onnx', 'model.onnx.data']) {
      await copyFile(join(src, f), join(dir, 'tampered', f));
    }
    const dataPath = join(dir, 'tampered', 'model.onnx.data');
    const bytes = await readFile(dataPath);
    bytes[100] ^= 0xff;
    await writeFile(dataPath, bytes);
    tamperedManifest = join(dir, 'tampered', 'manifest.json');
  });

  after(async () => {
    await rm(dir, { recursive: true, force: true });
  });

  it('starts with no environment override at all (npm run server)', async () => {
    // Previously this fell through to './model.onnx', which does not exist at
    // the repository root, so this path was simply broken.
    const { stdout } = await runNode('server/index.js', {
      env: { PORT: '0' },
      until: /Server running/,
    });
    assert.match(stdout, new RegExp(`Model id:\\s+${DEFAULT_MODEL_ID}`));
    assert.match(stdout, /source_checkpoint=unknown/);
  });

  it('resolves the same artifact when started from a foreign cwd', async () => {
    const { stdout } = await runNode(join(REPO_ROOT, 'server/index.js'), {
      cwd: tmpdir(),
      env: { PORT: '0' },
      until: /Server running/,
    });
    assert.match(stdout, new RegExp(`Model path: ${join(REPO_ROOT, 'models', DEFAULT_MODEL_ID)}`));
  });

  it('exits non-zero on a missing manifest', async () => {
    const { code, stderr } = await runNode('server/index.js', {
      env: { PORT: '0', MODEL_MANIFEST: join(dir, 'gone', 'manifest.json') },
    });
    assert.strictEqual(code, 1);
    assert.match(stderr, /MANIFEST_MISSING/);
  });

  it('exits non-zero on tampered weights', async () => {
    const { code, stderr } = await runNode('server/index.js', {
      env: { PORT: '0', MODEL_MANIFEST: tamperedManifest },
    });
    assert.strictEqual(code, 1);
    assert.match(stderr, /DATA_HASH_MISMATCH/);
  });

  it('leaves the pinned artifact untouched throughout', async () => {
    const { manifest, graphPath, dataPath } = await resolveModel({}, REPO_ROOT);
    assert.strictEqual(sha256(await readFile(graphPath)), manifest.graph.sha256);
    assert.strictEqual(sha256(await readFile(dataPath)), manifest.external_data.sha256);
  });
});

describe('launcher entry point', () => {
  const launcherEnv = { TWIXT_PORT: '0', TWIXT_AI_PORT: '0', TWIXT_NO_BROWSER: '1' };

  it('starts the AI server when the pinned artifact validates', async () => {
    const { stdout } = await runNode('scripts/startServer.js', {
      env: launcherEnv,
      until: /\[AI\]/,
      signal: 'SIGINT',
    });
    assert.match(stdout, new RegExp(`Model id: ${DEFAULT_MODEL_ID}`));
    assert.match(stdout, /\[AI\]/, 'AI child must have produced output');
  });

  it('refuses to start the AI server on a broken manifest, and exports nothing', async () => {
    const { stdout, stderr } = await runNode('scripts/startServer.js', {
      env: { ...launcherEnv, MODEL_MANIFEST: join(tmpdir(), 'definitely-absent', 'manifest.json') },
      until: /Press Ctrl\+C/,
      signal: 'SIGINT',
    });
    assert.match(stderr, /MODEL VALIDATION FAILED/);
    assert.match(stderr, /MANIFEST_MISSING/);
    assert.match(stderr, /No model was exported, substituted, or regenerated/);
    assert.match(stdout, /AI:\s+UNAVAILABLE/);
    assert.doesNotMatch(stdout, /\[AI\]/, 'no AI child may be spawned');
  });
});

describe('no startup export path', () => {
  it('the loader cannot spawn an exporter', async () => {
    // Constructed negatively: the previous design shelled out to an exporter
    // on startup. Assert the capability is absent from the module rather than
    // trusting that it is merely unused.
    const src = await readFile(join(HERE, 'model_manifest.js'), 'utf8');
    for (const forbidden of ['child_process', 'export_onnx', 'safetensors', 'checkpoints']) {
      assert.ok(!src.includes(`'${forbidden}`), `loader must not reference ${forbidden}`);
    }
    assert.ok(!/\bexec\(|\bspawn\(/.test(src), 'loader must not exec or spawn');
  });

  it('startServer.js no longer auto-exports the latest checkpoint', async () => {
    const src = await readFile(join(REPO_ROOT, 'scripts', 'startServer.js'), 'utf8');
    assert.ok(!src.includes('ensureOnnxModel'), 'auto-export entry point must be gone');
    assert.ok(!src.includes('findLatestCheckpoint'), 'lexicographic checkpoint pick must be gone');
    assert.ok(!src.includes('export_onnx'), 'startup export must be gone');
    assert.ok(!src.includes('MODEL_PATH'), 'MODEL_PATH must not be a serving override');
  });

  it('server/index.js takes its model from the manifest, not a cwd-relative default', async () => {
    const src = await readFile(join(REPO_ROOT, 'server', 'index.js'), 'utf8');
    assert.ok(!src.includes("'./model.onnx'"), 'cwd-relative default must be gone');
    assert.ok(!src.includes('MODEL_PATH'), 'MODEL_PATH must not be a serving override');
    assert.ok(src.includes('resolveModel'), 'must load through the manifest');
  });
});
