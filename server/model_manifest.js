/**
 * The single path by which a servable model is identified, validated and located.
 *
 * It replaces two unpinned seams that between them meant the product had no
 * pinned model at all:
 *
 *   1. The launcher re-exported the lexicographically last file from a research
 *      scratch directory whenever that file's mtime beat the ONNX artifact's.
 *      The served model was therefore decided by a directory listing and a
 *      timestamp race, not by a decision anyone made.
 *   2. The inference server defaulted to './model.onnx' — a cwd-relative path,
 *      so the served artifact depended on where node happened to be invoked.
 *
 * A served artifact is a PAIR: the ONNX graph plus its external-data sidecar,
 * which the graph references by relative filename from inside its own bytes.
 * Hashing the graph alone does not identify it; the graph here is 82,855 bytes
 * of topology pointing at 7,493,120 bytes of weights.
 *
 * Every failure in this module is fatal and carries a distinct code. There is
 * no fallback path and no startup export: silently substituting or
 * regenerating an artifact is the exact behaviour this module exists to end.
 *
 * A validated manifest asserts IDENTITY, not VALIDITY. It says these are the
 * bytes that were pinned. It makes no parity or playing-strength claim.
 */
import { readFile, stat } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { dirname, join, resolve, isAbsolute } from 'node:path';
import { fileURLToPath } from 'node:url';

import { NUM_CHANNELS, BOARD_SIZE } from './gameLogic.js';

const APP_ROOT = fileURLToPath(new URL('..', import.meta.url));

/**
 * The pinned baseline.
 *
 * Content-addressed over the WHOLE artifact: the id is a digest of both file
 * hashes, so it cannot collide across models that share a graph. Two exports
 * of different weights produce byte-identical graphs whenever the topology and
 * the external-data filenames match — an id derived from the graph alone would
 * name both.
 *
 * Changing the served model means editing this constant — a tracked, reviewable
 * diff. That is deliberate. It is not a runtime lookup, a newest-wins scan, or
 * an environment variable, because each of those is how the previous design
 * changed the served model without anyone deciding to.
 */
export const DEFAULT_MODEL_ID = 'c34b7ff3297c785a';
export const DEFAULT_MANIFEST_PATH = join(
  APP_ROOT,
  'models',
  DEFAULT_MODEL_ID,
  'manifest.json'
);

export const SUPPORTED_MANIFEST_VERSION = 1;

/** Dotted paths that must be present. Absence is a hard failure, not a default. */
const REQUIRED_FIELDS = [
  'manifest_version',
  'model_id',
  'description',
  'graph.filename',
  'graph.size_bytes',
  'graph.sha256',
  'external_data.filename',
  'external_data.size_bytes',
  'external_data.sha256',
  'provenance.source_checkpoint_path',
  'provenance.source_checkpoint_sha1',
  'provenance.export_commit',
  'provenance.exporter_config',
  'provenance.runtime_versions',
  'contract.inputs',
  'contract.outputs',
  'contract.board_shape',
  'contract.max_moves',
  'contract.onnxruntime_node',
  'claims',
];

export class ModelManifestError extends Error {
  constructor(code, message) {
    super(message);
    this.name = 'ModelManifestError';
    this.code = code;
  }
}

const fail = (code, message) => {
  throw new ModelManifestError(code, message);
};

const sha256 = (buf) => createHash('sha256').update(buf).digest('hex');

const getPath = (obj, dotted) =>
  dotted
    .split('.')
    .reduce((acc, key) => (acc == null ? undefined : acc[key]), obj);

/**
 * The artifact's identity: a digest over BOTH file hashes.
 *
 * Exposed so a staged artifact's id can be derived the same way the loader
 * checks it, rather than chosen by hand.
 */
export function computeModelId(graphSha256, dataSha256) {
  return createHash('sha256')
    .update(`${graphSha256}:${dataSha256}`)
    .digest('hex')
    .slice(0, 16);
}

// ---------------------------------------------------------------------------
// Minimal ONNX protobuf reader
//
// Only enough to answer one question: which external-data files does this graph
// actually reference? A substring search for the declared filename cannot
// answer it — any occurrence anywhere in the serialized bytes satisfies one,
// including a doc string, so a graph can name a decoy in its metadata while its
// initializers load something else entirely.
//
// Field numbers from the ONNX schema:
//   ModelProto.graph              = 7
//   GraphProto.initializer        = 5   (repeated TensorProto)
//   TensorProto.external_data     = 13  (repeated StringStringEntryProto)
//   StringStringEntryProto.key    = 1
//   StringStringEntryProto.value  = 2
// ---------------------------------------------------------------------------

/** The wire encoding of a StringStringEntryProto whose key is exactly "location". */
const ENCODED_LOCATION_KEY = Buffer.from([
  0x0a,
  0x08,
  ...Buffer.from('location', 'utf8'),
]);

function readVarint(buf, offset) {
  let result = 0;
  let shift = 0;
  let byte;
  let i = offset;
  do {
    if (i >= buf.length)
      fail('GRAPH_UNPARSEABLE', 'truncated varint in ONNX graph');
    byte = buf[i++];
    result += (byte & 0x7f) * 2 ** shift;
    shift += 7;
    if (shift > 63) fail('GRAPH_UNPARSEABLE', 'oversized varint in ONNX graph');
  } while (byte & 0x80);
  return [result, i];
}

/** Yield the top-level fields of one length-delimited protobuf message. */
function* protoFields(buf) {
  let i = 0;
  while (i < buf.length) {
    let tag;
    [tag, i] = readVarint(buf, i);
    const field = Math.floor(tag / 8);
    const wire = tag % 8;
    if (wire === 0) {
      let value;
      [value, i] = readVarint(buf, i);
      yield { field, wire, varint: value };
    } else if (wire === 1) {
      yield { field, wire };
      i += 8;
    } else if (wire === 2) {
      let len;
      [len, i] = readVarint(buf, i);
      if (i + len > buf.length)
        fail('GRAPH_UNPARSEABLE', 'truncated length-delimited field');
      yield { field, wire, bytes: buf.subarray(i, i + len) };
      i += len;
    } else if (wire === 5) {
      yield { field, wire };
      i += 4;
    } else {
      fail(
        'GRAPH_UNPARSEABLE',
        `unsupported protobuf wire type ${wire} in ONNX graph`
      );
    }
  }
}

/** Every external-data `location` reachable through ModelProto.graph.initializer. */
export function externalDataLocations(graphBytes) {
  let graph = null;
  for (const f of protoFields(graphBytes)) {
    if (f.field === 7 && f.wire === 2) graph = f.bytes;
  }
  if (graph === null)
    fail('GRAPH_UNPARSEABLE', 'ONNX file contains no ModelProto.graph');

  const locations = [];
  for (const g of protoFields(graph)) {
    if (g.field !== 5 || g.wire !== 2) continue; // GraphProto.initializer
    for (const t of protoFields(g.bytes)) {
      if (t.field !== 13 || t.wire !== 2) continue; // TensorProto.external_data
      let key = null;
      let value = null;
      for (const e of protoFields(t.bytes)) {
        if (e.field === 1 && e.wire === 2) key = e.bytes.toString('utf8');
        if (e.field === 2 && e.wire === 2) value = e.bytes.toString('utf8');
      }
      if (key === 'location') locations.push(value ?? '');
    }
  }
  return locations;
}

/** A filename that must stay inside the artifact directory. */
function assertPlainBasename(name, what) {
  if (
    name === '' ||
    name.includes('/') ||
    name.includes('\\') ||
    name.split('/').includes('..')
  ) {
    fail(
      'MANIFEST_INVALID',
      `${what} must be a plain filename, not a path: "${name}"`
    );
  }
}

/**
 * Bind the graph to exactly the declared sidecar.
 *
 * Three things must hold: the graph references external data at all, every
 * reference names the declared file, and no external-data entry hides outside
 * the initializers the structured walk covers. The last is checked by counting
 * the encoded "location" keys in the raw bytes and requiring the structured
 * walk to have reached all of them.
 */
function assertExternalDataBinding(graphBytes, graphPath, declaredFilename) {
  const locations = externalDataLocations(graphBytes);

  if (locations.length === 0) {
    fail(
      'EXTERNAL_REF_MISMATCH',
      `ONNX graph at ${graphPath} references no external data, but the manifest declares ` +
        `sidecar "${declaredFilename}"`
    );
  }

  let rawCount = 0;
  for (
    let i = 0;
    (i = graphBytes.indexOf(ENCODED_LOCATION_KEY, i)) !== -1;
    i += 1
  )
    rawCount++;
  if (rawCount !== locations.length) {
    fail(
      'EXTERNAL_REF_MISMATCH',
      `ONNX graph at ${graphPath} encodes ${rawCount} external-data location keys but only ` +
        `${locations.length} are reachable through graph.initializer; external data carried ` +
        `elsewhere in the graph is not supported`
    );
  }

  const distinct = [...new Set(locations)];
  if (distinct.length !== 1 || distinct[0] !== declaredFilename) {
    fail(
      'EXTERNAL_REF_MISMATCH',
      `ONNX graph at ${graphPath} loads external data from ` +
        `${distinct.map((d) => `"${d}"`).join(', ')}, but the manifest declares ` +
        `"${declaredFilename}"`
    );
  }
}

/**
 * Where to read the manifest from.
 *
 * The default is resolved against the application, never the current working
 * directory. An explicit MODEL_MANIFEST override is for staging and gets the
 * identical validation — it names a manifest, never a bare artifact, so there
 * is no way to serve unvalidated bytes by pointing at a file.
 */
export function resolveManifestPath(env = process.env, cwd = process.cwd()) {
  const override = env.MODEL_MANIFEST;
  if (!override) return DEFAULT_MANIFEST_PATH;
  return isAbsolute(override) ? override : resolve(cwd, override);
}

/** Read and structurally validate the manifest itself. Does not touch the artifact. */
export async function loadManifest(manifestPath) {
  let raw;
  try {
    raw = await readFile(manifestPath, 'utf8');
  } catch (err) {
    if (err.code === 'ENOENT') {
      fail('MANIFEST_MISSING', `No model manifest at ${manifestPath}`);
    }
    fail(
      'MANIFEST_UNREADABLE',
      `Cannot read model manifest ${manifestPath}: ${err.message}`
    );
  }

  let manifest;
  try {
    manifest = JSON.parse(raw);
  } catch (err) {
    fail(
      'MANIFEST_UNREADABLE',
      `Model manifest ${manifestPath} is not valid JSON: ${err.message}`
    );
  }

  if (manifest?.manifest_version !== SUPPORTED_MANIFEST_VERSION) {
    fail(
      'MANIFEST_INVALID',
      `Model manifest ${manifestPath} declares manifest_version ` +
        `${manifest?.manifest_version}; this build supports ${SUPPORTED_MANIFEST_VERSION}`
    );
  }

  const missing = REQUIRED_FIELDS.filter(
    (f) => getPath(manifest, f) === undefined
  );
  if (missing.length > 0) {
    fail(
      'MANIFEST_INVALID',
      `Model manifest ${manifestPath} is missing required field(s): ${missing.join(', ')}`
    );
  }

  assertPlainBasename(manifest.graph.filename, 'graph.filename');
  assertPlainBasename(
    manifest.external_data.filename,
    'external_data.filename'
  );

  // The id names the pair. A manifest whose id does not follow from its own
  // declared hashes is describing something other than what it points at.
  const expectedId = computeModelId(
    manifest.graph.sha256,
    manifest.external_data.sha256
  );
  if (manifest.model_id !== expectedId) {
    fail(
      'MODEL_ID_MISMATCH',
      `Model manifest ${manifestPath} declares model_id "${manifest.model_id}", but its ` +
        `graph and external-data hashes derive "${expectedId}"`
    );
  }

  return manifest;
}

/** Size + content check for one file of the pair. */
async function checkFile(path, declared, codes, label) {
  let info;
  try {
    info = await stat(path);
  } catch (err) {
    if (err.code === 'ENOENT')
      fail(codes.missing, `${label} not found at ${path}`);
    fail(codes.missing, `${label} at ${path} is unreadable: ${err.message}`);
  }

  if (info.size !== declared.size_bytes) {
    fail(
      codes.size,
      `${label} size mismatch at ${path}: manifest declares ${declared.size_bytes} bytes, ` +
        `found ${info.size}`
    );
  }

  const bytes = await readFile(path);
  const actual = sha256(bytes);
  if (actual !== declared.sha256) {
    fail(
      codes.hash,
      `${label} SHA-256 mismatch at ${path}: manifest declares ${declared.sha256}, ` +
        `found ${actual}`
    );
  }
  return bytes;
}

/**
 * Verify the artifact pair on disk against the manifest.
 *
 * Both files are resolved relative to the manifest's own directory, which is
 * also where the ONNX runtime resolves the sidecar from. Keeping all three in
 * one directory is what makes the sidecar's relative filename hold.
 */
export async function validateArtifact(manifest, manifestDir) {
  const graphPath = join(manifestDir, manifest.graph.filename);
  const dataPath = join(manifestDir, manifest.external_data.filename);

  const graphBytes = await checkFile(
    graphPath,
    manifest.graph,
    {
      missing: 'GRAPH_MISSING',
      size: 'GRAPH_SIZE_MISMATCH',
      hash: 'GRAPH_HASH_MISMATCH',
    },
    'ONNX graph'
  );

  assertExternalDataBinding(
    graphBytes,
    graphPath,
    manifest.external_data.filename
  );

  await checkFile(
    dataPath,
    manifest.external_data,
    {
      missing: 'DATA_MISSING',
      size: 'DATA_SIZE_MISMATCH',
      hash: 'DATA_HASH_MISMATCH',
    },
    'ONNX external data'
  );

  return { graphPath, dataPath };
}

const sameShape = (a, b) =>
  Array.isArray(a) &&
  Array.isArray(b) &&
  a.length === b.length &&
  a.every((v, i) => v === b[i]);

/**
 * The interface this server can actually consume — fixed by the application,
 * not by the artifact.
 *
 * `AlphaZeroInference.evaluate` feeds exactly these four tensors, builds the
 * board from NUM_CHANNELS and BOARD_SIZE, sizes the move buffers from
 * `maxMoves`, and reads `results.policy_logits.data` and `results.value.data`.
 * A model and a manifest that agree with each other can still describe an
 * interface none of that code can use, so agreement between them is not
 * sufficient.
 *
 * The engine itself supports other board sizes — curriculum training runs 8x8
 * up to 24x24, and `TwixtState` takes a size. The SERVED product is pinned to
 * the official 24x24, the only size this deployment cares about, so an artifact
 * built for another size is out of scope rather than impossible. Changing that
 * would mean changing BOARD_SIZE in gameLogic.js, which this contract follows
 * automatically; it must never become something the artifact gets to declare.
 *
 * NUM_CHANNELS is fixed by the trained architecture and has changed before
 * (24 pre-Phase-2, 30 after), which is exactly why it is read from the
 * application rather than accepted from the artifact.
 *
 * Shapes are matched exactly. The value head is read as `data[0]`, so a `[1]`
 * output would also function; requiring the scalar `[]` this exporter produces
 * keeps the rule uniform, and widening it should be a deliberate edit here
 * rather than a surprise at startup.
 */
export function applicationContract(maxMoves) {
  return {
    inputs: [
      {
        name: 'board',
        type: 'float32',
        shape: [1, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE],
      },
      { name: 'move_rows', type: 'int64', shape: [maxMoves] },
      { name: 'move_cols', type: 'int64', shape: [maxMoves] },
      { name: 'move_mask', type: 'float32', shape: [maxMoves] },
    ],
    outputs: [
      { name: 'policy_logits', type: 'float32', shape: [maxMoves] },
      { name: 'value', type: 'float32', shape: [] },
    ],
  };
}

/** Check a declared contract against what this server's code can consume. */
function assertApplicationContract(contract, wrapperMaxMoves) {
  const required = applicationContract(wrapperMaxMoves);

  const compare = (declaredList, requiredList, what) => {
    const declared = new Map(declaredList.map((t) => [t.name, t]));

    for (const req of requiredList) {
      const got = declared.get(req.name);
      if (!got) {
        fail(
          'APPLICATION_MISMATCH',
          `this server feeds and reads ${what} "${req.name}", which the artifact does not ` +
            `provide; it provides [${[...declared.keys()].join(', ')}]`
        );
      }
      if (got.type !== req.type) {
        fail(
          'APPLICATION_MISMATCH',
          `this server supplies ${what} "${req.name}" as ${req.type}, but the artifact ` +
            `declares ${got.type}`
        );
      }
      if (!sameShape(got.shape, req.shape)) {
        fail(
          'APPLICATION_MISMATCH',
          `this server supplies ${what} "${req.name}" with shape [${req.shape}] ` +
            `(from NUM_CHANNELS=${NUM_CHANNELS}, BOARD_SIZE=${BOARD_SIZE}, ` +
            `maxMoves=${wrapperMaxMoves}), but the artifact declares [${got.shape}]`
        );
      }
    }

    const extra = [...declared.keys()].filter(
      (n) => !requiredList.some((r) => r.name === n)
    );
    if (extra.length > 0) {
      fail(
        'APPLICATION_MISMATCH',
        `artifact declares ${what}(s) this server does not handle: ${extra.join(', ')}`
      );
    }
  };

  compare(contract.inputs, required.inputs, 'input');
  compare(contract.outputs, required.outputs, 'output');
}

/**
 * Check the loaded model three ways.
 *
 *   1. manifest vs session — does the artifact match what the manifest says?
 *   2. manifest internal consistency — do board_shape and max_moves follow
 *      from the tensors declared beside them?
 *   3. manifest vs application — can this server's code consume it at all?
 *
 * The third is not implied by the first two. A model and a manifest can agree
 * perfectly on a 24-channel board or an output named `score`, and this server
 * would still feed 30 channels and read `results.value`. Because step 1 forces
 * the session to equal the manifest, checking the manifest against the fixed
 * application contract also binds the session to it.
 *
 * `wrapperMaxMoves` is AlphaZeroInference.maxMoves. It sizes the move buffers
 * and caps the policy read; if it disagrees with the artifact, priors are read
 * past the end of the output or legal moves are silently dropped. It is
 * required — there is no version of this check that is safe to skip.
 */
export function assertSessionContract(manifest, session, wrapperMaxMoves) {
  const { contract } = manifest;

  if (!Number.isInteger(wrapperMaxMoves)) {
    fail(
      'APPLICATION_MISMATCH',
      `the application's max-move count is required to validate a model, got ${wrapperMaxMoves}`
    );
  }

  const check = (actualList, declaredList, what) => {
    const actual = new Map(actualList.map((m) => [m.name, m]));
    const declared = new Map(declaredList.map((m) => [m.name, m]));

    const actualNames = [...actual.keys()].sort();
    const declaredNames = [...declared.keys()].sort();
    if (
      actualNames.length !== declaredNames.length ||
      actualNames.some((n, i) => n !== declaredNames[i])
    ) {
      fail(
        'CONTRACT_MISMATCH',
        `Model ${what} do not match the manifest contract: expected ` +
          `[${declaredNames.join(', ')}], loaded [${actualNames.join(', ')}]`
      );
    }

    for (const [name, spec] of declared) {
      const live = actual.get(name);
      if (live.type !== spec.type) {
        fail(
          'CONTRACT_MISMATCH',
          `Model ${what} "${name}" has type ${live.type}, manifest declares ${spec.type}`
        );
      }
      if (!sameShape(live.shape, spec.shape)) {
        fail(
          'CONTRACT_MISMATCH',
          `Model ${what} "${name}" has shape [${live.shape}], manifest declares [${spec.shape}]`
        );
      }
    }
  };

  // 1. Is the artifact what the manifest says it is?
  check(session.inputMetadata, contract.inputs, 'inputs');
  check(session.outputMetadata, contract.outputs, 'outputs');

  // 2. The two semantic fields the rest of the server is built from must
  // follow from the tensors, not merely sit beside them.
  const board = contract.inputs.find((t) => t.name === 'board');
  if (!sameShape(contract.board_shape, board?.shape)) {
    fail(
      'CONTRACT_MISMATCH',
      `Manifest board_shape [${contract.board_shape}] does not match its declared ` +
        `board input shape [${board?.shape}]`
    );
  }

  for (const name of ['move_rows', 'move_cols', 'move_mask', 'policy_logits']) {
    const tensor =
      contract.inputs.find((t) => t.name === name) ??
      contract.outputs.find((t) => t.name === name);
    if (tensor?.shape?.[0] !== contract.max_moves) {
      fail(
        'CONTRACT_MISMATCH',
        `Manifest max_moves ${contract.max_moves} does not match declared "${name}" ` +
          `shape [${tensor?.shape}]`
      );
    }
  }

  // 3. Can this server actually consume it?
  if (wrapperMaxMoves !== contract.max_moves) {
    fail(
      'APPLICATION_MISMATCH',
      `AlphaZeroInference.maxMoves is ${wrapperMaxMoves} but the artifact provides ` +
        `${contract.max_moves}; move buffers and the policy read would disagree with the model`
    );
  }
  assertApplicationContract(contract, wrapperMaxMoves);
}

/**
 * Resolve, load and validate in one step. Throws on any failure; never returns
 * a partially trusted result.
 */
export async function resolveModel(env = process.env, cwd = process.cwd()) {
  const manifestPath = resolveManifestPath(env, cwd);
  const manifest = await loadManifest(manifestPath);
  const { graphPath, dataPath } = await validateArtifact(
    manifest,
    dirname(manifestPath)
  );
  return { manifestPath, manifest, graphPath, dataPath };
}
