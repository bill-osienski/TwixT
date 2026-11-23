/**
 * The hard model's own selection path (integration proposal §4.7) and the
 * retained rollback pins (§8). Manifest resolution and file hashing only: no
 * ONNX session is created here.
 *
 * Also the H-MX integration's packaging check (proposal
 * docs/superpowers/2026-11-22-htb-mix-integration-proposal.md, revision 1, B1):
 * models/627c8d774df838a8/ holds the exact pair the export and match records
 * qualify, under a manifest pinned by its own hash.
 */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { cp, mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
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

  it('the pinned hard model is the student export qualified by V2-R', () => {
    assert.equal(HARD_MODEL_ID, '78c16a31c3dba37f');
    assert.equal(resolveHardManifestPath({}), join(REPO_ROOT, 'models', '78c16a31c3dba37f', 'manifest.json'));
  });

  it('an unpinned hard path refuses with a distinct code', () => {
    assert.throws(
      () => resolveHardManifestPath({}, process.cwd(), null),
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
    assert.equal(resolveHardManifestPath({ MODEL_MANIFEST: A.path }), resolveHardManifestPath({}));
    assert.throws(() => resolveHardManifestPath({ MODEL_MANIFEST: A.path }, process.cwd(), null), (e) => e.code === 'HARD_MODEL_UNPINNED');
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

describe('the retained rollback pins and the pinned hard model are byte-identical (§8)', () => {
  for (const id of [DEFAULT_MODEL_ID, '1d64027db521a50f', '78c16a31c3dba37f', '627c8d774df838a8', HARD_MODEL_ID]) {
    it(`models/${id}: the manifest validates and both files hash to its pair`, async () => {
      const manifestPath = join(REPO_ROOT, 'models', id, 'manifest.json');
      const manifest = await loadManifest(manifestPath);
      assert.equal(manifest.model_id, id);
      await validateArtifact(manifest, dirname(manifestPath)); // sizes and SHA-256 of graph and sidecar
    });
  }
  it('the pinned hard model resolves through its own path with no environment override', async () => {
    const hard = await resolveHardModel({});
    assert.equal(hard.manifest.model_id, '78c16a31c3dba37f');
    assert.equal(hard.manifest.provenance.source_checkpoint_sha256, '0e1cb6211aa0b3e2dec6ab6f39bec74ceb8c472c7dcf603e7d331b1d1c1c34e5');
  });
  it('the served default is still c34b7ff3297c785a', () => {
    assert.equal(DEFAULT_MODEL_ID, 'c34b7ff3297c785a');
  });
});

// --- H-MX integration, B1: the tracked artifact is the qualified pair --------
//
// Each hash is checked against the committed record that holds it. The graph
// and data hashes are in both the export record (attempt 2) and the match
// record. Neither record holds a manifest hash or the file sizes: the manifest
// is pinned here by its own sha256 for packaging, and what makes it the tested
// artifact is that it names the qualified pair (proposal revision 1, §2, B1).

const HMX_ID = '627c8d774df838a8';
const HMX_MANIFEST_SHA256 = 'abf3931d920f788de146cf8dafbd213581f682a754b65df1520d42eae954071d';
const HMX_EXPORT_RECORD = 'docs/superpowers/evidence/htb-mix-export-2/record.json';
const HMX_MATCH_RECORD = 'docs/superpowers/evidence/htb-mix-match/record.json';
const HMX_EXPORT_DIR = join(process.env.HOME ?? '', 'twixt-cache', 'htb-mix', 'export', HMX_ID);

/** Every way the directory fails to be the qualified pair; empty when it is. */
async function qualifiedPairProblems(dir, exportRec, matchRec) {
  const p = [];
  const ex = exportRec.challenger;
  const mx = matchRec.challenger;
  for (const [k, a, b] of [
    ['model_id', ex.model_id, mx.model_id],
    ['graph_sha256', ex.graph_sha256, mx.graph_sha256],
    ['external_data_sha256', ex.external_data_sha256, mx.external_data_sha256],
  ]) {
    if (a !== b) p.push(`the records disagree on ${k}`);
  }
  if (ex.model_id !== HMX_ID) p.push(`the records qualify ${ex.model_id}, not ${HMX_ID}`);
  const raw = await readFile(join(dir, 'manifest.json'));
  if (sha256(raw) !== HMX_MANIFEST_SHA256) p.push('manifest.json is not the pinned bytes');
  let m;
  try {
    m = JSON.parse(raw.toString('utf8'));
  } catch {
    return [...p, 'manifest.json is unparseable'];
  }
  if (m.model_id !== ex.model_id) p.push('manifest model_id is not the qualified id');
  if (m.graph?.filename !== 'model.onnx') p.push('manifest graph.filename is not model.onnx');
  if (m.graph?.sha256 !== ex.graph_sha256) p.push('manifest graph.sha256 is not the qualified graph');
  if (m.external_data?.filename !== 'model.onnx.data') p.push('manifest external_data.filename is not model.onnx.data');
  if (m.external_data?.sha256 !== ex.external_data_sha256) p.push('manifest external_data.sha256 is not the qualified data');
  if (computeModelId(m.graph?.sha256 ?? '', m.external_data?.sha256 ?? '') !== HMX_ID) p.push('the manifest pair does not compute to the id');
  const src = m.provenance?.source_checkpoint_sha256;
  if (src !== exportRec.admission?.challenger?.sha256 || src !== mx.source_sha256) p.push('manifest source checkpoint is not the recorded one');
  for (const [f, want] of [['model.onnx', ex.graph_sha256], ['model.onnx.data', ex.external_data_sha256]]) {
    if (sha256(await readFile(join(dir, f))) !== want) p.push(`${f} is not the qualified bytes`);
  }
  try {
    await validateArtifact(m, dir); // sizes and both hashes against the manifest
  } catch (e) {
    p.push(`validateArtifact: ${e.code ?? e.message}`);
  }
  return p;
}

describe('H-MX integration B1: models/627c8d774df838a8 is the qualified pair', () => {
  const dir = join(REPO_ROOT, 'models', HMX_ID);
  let exportRec;
  let matchRec;
  let tmp;
  before(async () => {
    exportRec = JSON.parse(await readFile(join(REPO_ROOT, HMX_EXPORT_RECORD), 'utf8'));
    matchRec = JSON.parse(await readFile(join(REPO_ROOT, HMX_MATCH_RECORD), 'utf8'));
    tmp = await mkdtemp(join(tmpdir(), 'twixt-hmx-b1-'));
  });
  after(async () => rm(tmp, { recursive: true, force: true }));

  it('the records qualify 627c8d774df838a8 by graph and data hash, and agree', () => {
    assert.equal(exportRec.status, 'COMPLETED');
    assert.equal(exportRec.verdict, 'PARITY_PASS');
    assert.equal(exportRec.challenger.graph_sha256, '3beaeed15a97f28c69e639b6c5ab2466f961a1aa75f232755e581d592fcb55af');
    assert.equal(exportRec.challenger.external_data_sha256, 'cf78b17f7e9368e5be850ded8b41d3c13921f32bf54c3fab0273cfcb8489822f');
    assert.equal(matchRec.status, 'COMPLETED');
    assert.deepEqual(
      [matchRec.challenger.model_id, matchRec.challenger.graph_sha256, matchRec.challenger.external_data_sha256],
      [exportRec.challenger.model_id, exportRec.challenger.graph_sha256, exportRec.challenger.external_data_sha256]
    );
    // Neither record carries a manifest hash: the pin below is a packaging pin.
    assert.ok(!JSON.stringify(exportRec).includes(HMX_MANIFEST_SHA256));
    assert.ok(!JSON.stringify(matchRec).includes(HMX_MANIFEST_SHA256));
  });

  it('the tracked directory is the qualified pair under the pinned manifest, and loads through the product path', async () => {
    assert.deepEqual(await qualifiedPairProblems(dir, exportRec, matchRec), []);
    const manifest = await loadManifest(join(dir, 'manifest.json'));
    assert.equal(manifest.model_id, HMX_ID);
    const hard = await resolveHardModel({ HARD_MODEL_MANIFEST: join(dir, 'manifest.json') });
    assert.equal(hard.manifest.model_id, HMX_ID);
  });

  it('the tracked files are byte copies of the export directory, when it is present', async (t) => {
    try {
      await readFile(join(HMX_EXPORT_DIR, 'manifest.json'));
    } catch {
      return t.skip(`${HMX_EXPORT_DIR} absent`);
    }
    for (const f of ['model.onnx', 'model.onnx.data', 'manifest.json']) {
      assert.equal(sha256(await readFile(join(dir, f))), sha256(await readFile(join(HMX_EXPORT_DIR, f))), f);
    }
  });

  // Controls: each tamper is refused, against the untampered copy above.
  const TAMPERS = {
    'one manifest byte': async (d) => {
      const raw = await readFile(join(d, 'manifest.json'));
      raw[raw.length - 2] ^= 0x01;
      await writeFile(join(d, 'manifest.json'), raw);
    },
    'the graph hash field': async (d) => {
      const m = JSON.parse(await readFile(join(d, 'manifest.json'), 'utf8'));
      m.graph.sha256 = 'f'.repeat(64);
      await writeFile(join(d, 'manifest.json'), JSON.stringify(m, null, 2));
    },
    'the data filename': async (d) => {
      const m = JSON.parse(await readFile(join(d, 'manifest.json'), 'utf8'));
      m.external_data.filename = 'other.onnx.data';
      await writeFile(join(d, 'manifest.json'), JSON.stringify(m, null, 2));
    },
    'one data byte': async (d) => {
      const raw = await readFile(join(d, 'model.onnx.data'));
      raw[0] ^= 0x01;
      await writeFile(join(d, 'model.onnx.data'), raw);
    },
  };
  for (const [name, tamper] of Object.entries(TAMPERS)) {
    it(`a tampered copy is refused: ${name}`, async () => {
      const d = join(tmp, name.replace(/\W+/g, '_'));
      await cp(dir, d, { recursive: true });
      assert.deepEqual(await qualifiedPairProblems(d, exportRec, matchRec), []); // the copy itself is clean
      await tamper(d);
      assert.ok((await qualifiedPairProblems(d, exportRec, matchRec)).length > 0, name);
    });
  }

  it('a record that names another pair is refused', async () => {
    const other = structuredClone(matchRec);
    other.challenger.graph_sha256 = 'e'.repeat(64);
    assert.ok((await qualifiedPairProblems(dir, exportRec, other)).length > 0);
  });
});
