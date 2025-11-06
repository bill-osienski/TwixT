// assets/js/ai/valueModel_v2.js
// Value model loader + scorer with optional standardization (mu/std) support.
// Works in browser and Node. Exposes:
//   - ensureValueModelLoaded(pathOrUrl?)
//   - isModelLoaded()
//   - evaluateValueModel(heuristics, featureContext)
//   - getLoadedModel()

/* Expected model JSON shape (from train_value.py output):
{
  "type": "logistic_regression",
  "generatedAt": "...",
  "feature_keys": [ ... order of features ... ],
  "weights": [bias, w1, w2, ...],
  "preproc": {
    "standardize": true|false,
    "mean": [ ... same length as feature_keys ... ] | null,
    "std":  [ ... same length as feature_keys ... ] | null
  },
  "params": {...},
  "metrics": {...}
}
*/

const isBrowser =
  typeof window !== 'undefined' && typeof window.document !== 'undefined';

// Where to look by default
const DEFAULT_BROWSER_PATH = '/assets/value-model.json';
const NODE_CANDIDATES = [
  'value-model.json', // project root
  'assets/value-model.json', // assets folder
];

let _model = null;
let _lastSource = null;

// ---------- I/O helpers ----------
async function readTextNode(path) {
  const fs = await import('fs/promises');
  return fs.readFile(path, 'utf8');
}

async function pathExistsNode(path) {
  const fs = await import('fs/promises');
  try {
    await fs.stat(path);
    return true;
  } catch {
    return false;
  }
}

async function fetchTextBrowser(url) {
  const res = await fetch(url, { cache: 'no-cache' });
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
  return res.text();
}

async function readModelSource(pathOrUrl) {
  if (isBrowser) {
    const src = pathOrUrl || DEFAULT_BROWSER_PATH;
    const txt = await fetchTextBrowser(src);
    return JSON.parse(txt);
  }

  // Node:
  if (pathOrUrl) {
    if (/^https?:\/\//i.test(pathOrUrl)) {
      const { default: fetch } = await import('node-fetch');
      const res = await fetch(pathOrUrl);
      if (!res.ok) throw new Error(`HTTP ${res.status} for ${pathOrUrl}`);
      return await res.json();
    } else {
      const txt = await readTextNode(pathOrUrl);
      return JSON.parse(txt);
    }
  }

  // Try common local paths
  for (const candidate of NODE_CANDIDATES) {
    if (await pathExistsNode(candidate)) {
      const txt = await readTextNode(candidate);
      return JSON.parse(txt);
    }
  }

  // Last resort: browser-style default (useful if running under a server)
  try {
    const txt = await readTextNode(DEFAULT_BROWSER_PATH.replace(/^\//, '')); // "assets/value-model.json"
    return JSON.parse(txt);
  } catch {
    throw new Error('value-model.json not found in project root or assets/');
  }
}

// ---------- math helpers ----------
function sigmoid(z) {
  if (z < -35) return 1e-15;
  if (z > 35) return 1 - 1e-15;
  return 1 / (1 + Math.exp(-z));
}

// Aligns a flat numeric vector to model weights; applies standardization if present.
function applyPreproc(x, preproc) {
  if (!preproc || !preproc.standardize) return x;
  const mu = preproc.mean || null;
  const sd = preproc.std || null;
  if (!mu || !sd || mu.length !== x.length || sd.length !== x.length) {
    // Model says standardize but mu/std are missing or wrong length—fallback safely
    return x;
  }
  const out = new Array(x.length);
  for (let i = 0; i < x.length; i++) {
    const denom = sd[i] === 0 ? 1 : sd[i];
    out[i] = (x[i] - mu[i]) / denom;
  }
  return out;
}

// Build feature vector in the order the model expects.
function buildFeatureVector(model, heuristics = {}, featureContext = {}) {
  const keys = model.feature_keys || [];
  const x = new Array(keys.length);
  for (let i = 0; i < keys.length; i++) {
    const k = keys[i];
    // numbers may arrive as strings; coerce safely
    let v = heuristics[k] ?? featureContext[k] ?? 0;
    if (v == null || Number.isNaN(+v)) v = 0;
    x[i] = +v;
  }
  return x;
}

// ---------- public API ----------
export async function ensureValueModelLoaded(pathOrUrl) {
  if (_model) return _model;
  const payload = await readModelSource(pathOrUrl);
  validateModel(payload);
  _model = payload;
  _lastSource = pathOrUrl || '(auto)';
  return _model;
}

export function isModelLoaded() {
  return !!_model;
}

export function getLoadedModel() {
  return _model;
}

// Optional, lazy loader used by search.js. Safe if the file is missing.
let _loadPromise = null;
export async function maybeLoadValueModel(
  pathOrUrl = '/assets/value-model.json'
) {
  if (isModelLoaded()) return getLoadedModel();
  if (_loadPromise) return _loadPromise;

  _loadPromise = ensureValueModelLoaded(pathOrUrl)
    .catch(() => null) // swallow errors so human-vs-human still works
    .finally(() => {
      _loadPromise = null;
    });

  return _loadPromise;
}

// Returns { probability, logit, adjustment }.
// - probability: P(win | features)
// - logit: bias + w·x  (before sigmoid)
// - adjustment: probability - 0.5 (a centered “nudge” if you want one)
export function evaluateValueModel(heuristics = {}, featureContext = {}) {
  if (!_model) {
    return { probability: null, logit: null, adjustment: null };
  }
  const vec = buildFeatureVector(_model, heuristics, featureContext);
  const x = applyPreproc(vec, _model.preproc || null);

  // weights shape: [bias, w1, w2, ...]
  const w = _model.weights || [];
  if (!Array.isArray(w) || w.length !== x.length + 1) {
    // Shape mismatch—fail safely
    return { probability: null, logit: null, adjustment: null };
  }
  let z = w[0]; // bias
  for (let i = 0; i < x.length; i++) {
    z += w[i + 1] * x[i];
  }
  const p = sigmoid(z);
  return { probability: p, logit: z, adjustment: p - 0.5 };
}

// ---------- validation ----------
function validateModel(m) {
  if (!m || typeof m !== 'object') {
    throw new Error('Invalid value model: not an object.');
  }
  if (!Array.isArray(m.weights) || m.weights.length < 2) {
    throw new Error("Invalid value model: missing usable 'weights'.");
  }
  if (
    !Array.isArray(m.feature_keys) ||
    m.feature_keys.length !== m.weights.length - 1
  ) {
    throw new Error(
      "Invalid value model: 'feature_keys' must align with weights (minus bias)."
    );
  }
  // Optional preproc fields are fine; if present, lengths should match feature_keys.
  if (m.preproc && m.preproc.standardize) {
    const mu = m.preproc.mean;
    const sd = m.preproc.std;
    if (
      !Array.isArray(mu) ||
      !Array.isArray(sd) ||
      mu.length !== m.feature_keys.length ||
      sd.length !== m.feature_keys.length
    ) {
      throw new Error(
        'Invalid value model: preproc.mean/std must match feature_keys length.'
      );
    }
  }
}

export default {
  ensureValueModelLoaded,
  isModelLoaded,
  getLoadedModel,
  maybeLoadValueModel,
  evaluateValueModel,
};
