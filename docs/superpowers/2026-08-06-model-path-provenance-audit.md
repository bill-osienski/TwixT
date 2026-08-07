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
