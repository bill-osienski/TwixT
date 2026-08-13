# `MODEL_PATH` Provenance Audit — Read-Only

**Date:** 2026-08-06 · **Verdict: `MISMATCH`** · **Read-only; nothing was exported,
converted, replaced or modified.**

Recorded here rather than in the frozen design spec: this is an observation about the
product server, not a change to the experiment's thresholds, population or policy.

## Question

Does the ONNX artifact served by the product come from `calib020_0001`, the checkpoint
the competitive-readout experiment holds fixed?

## Verdict

**`MISMATCH`.** The served artifact cannot have come from `calib020_0001`, and the
evidence identifies a different, probable source.

## Selection chain

```
scripts/startServer.js:114   MODEL_PATH = <ROOT>/server/model.onnx
server/index.js:573          modelPath = process.env.MODEL_PATH || './model.onnx'
server/index.js:585          new AlphaZeroInference(modelPath)
```

`ensureOnnxModel` (`startServer.js:45-99`) re-exports whenever the newest checkpoint is
newer than the ONNX. Its source directory is hardcoded to
**`checkpoints/alphazero-v2-staged`** — not the calibration line —
and `findLatestCheckpoint` (`:31-40`) picks the lexicographically last `.safetensors`,
i.e. the highest iteration.

## The artifact

| | |
|---|---|
| path | `server/model.onnx` |
| sha256 | `f1b4411a9d46cc767aa31a3f6885c307704897f21c327a3210da5d5c810a6ae5` |
| size | 82,855 bytes |
| mtime | **2026-05-15 20:57:44** · birth 2026-04-23 22:13:23 |
| producer | `pytorch 2.10.0`, opset 18, IR version 10 |
| `metadata_props` | **NONE** |
| `doc_string` | empty (model and graph) |
| input | `board [1, 30, 24, 24]` |
| git | **untracked** — `.gitignore:53` `*.onnx` |

## Why `calib020_0001` is conclusively excluded

`checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors`
(sha1 `209cf2d4fd24a48553d259dd71b4954867b9473e`) was created **2026-06-20 22:14:54**.
The served ONNX was written **2026-05-15 20:57:44** — five weeks earlier.

An artifact cannot be exported from a checkpoint that does not yet exist. This holds
regardless of the missing provenance record, and needed no cross-format comparison.

## Probable source — CIRCUMSTANTIAL, not proven

`checkpoints/alphazero-v2-staged/model_iter_0193.safetensors`, mtime
**2026-05-15 20:51:45** — six minutes before the export, and the only checkpoint written
in that window. It was the highest-numbered file in the auto-export directory at that
moment, which is what `findLatestCheckpoint` selects, and its appearance is what would
have tripped the "checkpoint newer than ONNX" re-export.

**This identification is timestamp-plus-mechanism evidence and must stay labelled
circumstantial.** The *exclusion* above is the conclusive part, and it is the only part
the decision rests on. A weight-level comparison could settle the positive
identification; it was judged unnecessary and was not performed.

## Why no provenance record exists

- `export_onnx.py` writes no `metadata_props`, no `doc_string`, no source path.
- The artifact is gitignored, so there is no commit history for it.
- Documentation disagrees with the code: `README.md:38` exports from
  `checkpoints/alphazero-fresh/model_iter_0168.safetensors`, while `startServer.js` uses
  `alphazero-v2-staged`. **Neither is the calibration line.**

## Consequences, frozen

- **Phase A is a defect-repair workstream only and carries NO strength claim.** The gap
  is larger than "unverified": the product serves a checkpoint from a different training
  line, roughly 200 iterations before the staged run finished.
- The three server defects it fixed — divergent REST/WS readout policy, unreachable
  `deterministicMode`, and an unscoped cache — stand on their own merits.
- **Do not replace or re-export `model.onnx`** on the basis of this audit.
- ONNX source-metadata improvements are deferred to a **separate product-provenance
  task**. The smallest durable fix is for `export_onnx.py` to write the source checkpoint
  path and its SHA-1 into `metadata_props`, making future artifacts self-identifying. It
  cannot retroactively identify the current file.

---

## Erratum — 2026-08-13: the artifact is a PAIR, not one file

**The verdict is unaffected. The artifact description above is incomplete.**

The table in *The artifact* records `server/model.onnx` — 82,855 bytes, sha256
`f1b4411a…` — as though that file were the served artifact. It is only the **graph**.
The graph holds **33 externally stored initializers** referencing a sibling weight
sidecar by relative filename; the weights are not in it.

| | graph | external data |
|---|---|---|
| filename | `model.onnx` | `model.onnx.data` |
| size | 82,855 bytes | 7,493,120 bytes |
| sha256 | `f1b4411a9d46cc767aa31a3f6885c307704897f21c327a3210da5d5c810a6ae5` | `111546445ea4db8eb775adb7ca611539ac60c63780e200fb9a8ec861ab3b0937` |

Consequences of the correction:

- **The `MISMATCH` verdict stands unchanged.** Both files carry the same
  2026-05-15 20:57 mtime and both predate `calib020_0001` (2026-06-20 22:14:54). The
  exclusion never depended on the artifact's internal structure.
- **A single-file hash cannot identify this artifact.** Any check that hashes only the
  graph would pass while the weights were replaced wholesale.
- The probable-source identification remains **circumstantial** and was not pursued.
  Note additionally that BatchNorm is folded into the convolutions in this export, so a
  weight-level comparison against a `.safetensors` checkpoint would have to reproduce
  the fusion as well as the layout conversion. It has no decision value.

Measured directly from the artifact by walking the protobuf (`externalDataLocations`
in `server/model_manifest.js`), replacing an earlier estimate in this erratum that was
inferred from raw string counts and was wrong about the initializer arithmetic:

| | |
|---|---|
| nodes | 76 |
| initializers | 53, of which **33 carry external data** |
| external-data locations | 33, all exactly `model.onnx.data` |
| op types | `Relu` 16, `Conv` 14, `Slice` 8, `Add` 6, `Concat` 4, `Where` 4, `Gemm` 4, `Transpose` 2, `Gather` 2, `Unsqueeze` 2, `Squeeze` 2, `Tanh` 2, and nine singletons |
| `BatchNormalization` | **0** — folded |

**Superseded by Phase 1 (2026-08-13):** the selection chain recorded above no longer
exists. `ensureOnnxModel`'s auto-export and `findLatestCheckpoint`'s lexicographic pick
have been removed, and `server/index.js`'s cwd-relative `./model.onnx` default is gone.
The same pair, byte-identical and hash-verified across the move, now lives at
`models/1d64027db521a50f/` — content-addressed over **both** hashes, since an address
derived from the graph alone would collide across every export of this architecture —
with a manifest recording both hashes and `source_checkpoint: unknown`. Loading goes
through `server/model_manifest.js` for both entry points, binds the graph to its sidecar
by parsing the external-data references rather than searching for the filename, enforces
the tensor contract against runtime metadata, and fails loudly rather than exporting. See
`docs/superpowers/2026-08-13-product-model-alignment-decision-memo.md`.
