# Served model switched to the candidate

**Date:** 2026-08-22 · **Change: one constant.** `server/model_manifest.js`

```
-export const DEFAULT_MODEL_ID = '1d64027db521a50f';
+export const DEFAULT_MODEL_ID = 'c34b7ff3297c785a';
```

## Before and after

| | before (now rollback) | after (served) |
|---|---|---|
| `model_id` | `1d64027db521a50f` | `c34b7ff3297c785a` |
| graph sha256 | `f1b4411a9d46cc767aa31a3f6885c307704897f21c327a3210da5d5c810a6ae5` | `9df19e08ca438acbc3ba14d50925298389ddaf0d522f623fab617abdf24864ad` |
| graph bytes | 82,855 | 82,855 |
| data sha256 | `111546445ea4db8eb775adb7ca611539ac60c63780e200fb9a8ec861ab3b0937` | `fc1ffaacf397ebbef530ba2c7bbc76e092c549e4ebf2c7fa592508f50e4b64e2` |
| data bytes | 7,493,120 | 7,493,120 |
| source checkpoint | **unknown** | `checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors` |
| checkpoint sha1 | unknown | `209cf2d4fd24a48553d259dd71b4954867b9473e` |

**Both artifact pairs are byte-identical to before this change** — nothing was regenerated,
re-exported or moved. `models/1d64027db521a50f/` is retained in full as the rollback target. All
four hashes above were verified on disk against their manifests before the switch.

## Rollback

Revert the one constant:

```
sed -i '' "s/DEFAULT_MODEL_ID = 'c34b7ff3297c785a'/DEFAULT_MODEL_ID = '1d64027db521a50f'/" \
  server/model_manifest.js
```

Then confirm the banner reports `Model id: 1d64027db521a50f` and
`source_checkpoint=unknown`. No data migration, cache invalidation or artifact restoration is
involved: the baseline pair is still present and unmodified, and the loader validates whichever
pair the constant names by size and SHA-256 before serving it.

## Why the switch is a diff and not a lookup

`DEFAULT_MODEL_ID` is a tracked constant precisely so this is reviewable. The original defect was
that the served artifact was re-derived at every launch — a newest-wins scan — so it could change
without anyone deciding to. Changing what is served still means editing this line.

## Evidence boundary — what this rests on, and what it does not

**It rests on:** a preregistered 200-game match in the product's own stack (`aca5ca2`), analysed
once by the frozen §6 analyser (`303159f`). `ACCEPTED` / `CANDIDATE_STRONGER`: mean pair score
`0.8475`, bootstrap 95% `[0.7975, 0.8950]`, `t` 95% `[0.7962, 0.8988]`, both methods agreeing,
tally 76 win / 20 draw / 4 loss over 100 pairs. Also on Phase 2's parity PASS and on the
candidate's complete provenance against the incumbent's unknown provenance.

**It does NOT rest on, and this may not be claimed:**

- **Medium was never measured.** `medium` is `DEFAULT_DIFFICULTY`. §5.2 is explicit that a
  hard-arm result cannot support a claim about the default user experience, and it is not being
  used as one. The supported statement is **"decisively stronger at hard"**, not "the default
  medium experience was measured as better". Medium remains blocked on seeded RNG, since
  `server/mcts.js` calls bare `Math.random()` in `selectMove`'s stochastic branch.
- **No external anchor.** The result is relative: the candidate beats *these served bytes* in
  *this stack*. It says nothing about absolute strength.
- **No claim about the incumbent's identity.** What produced `1d64027db521a50f` is still unknown
  and was deliberately not pursued.

The switch was approved on that basis, with medium evidence explicitly not required first,
because the hard effect is large, parity is exact on the frozen corpus, provenance is complete,
and the change is reversible by reverting one line.

## Consequence for the execution surface

`server/model_manifest.js` is one of the ten execution-surface files, so this commit **moves the
surface digest off `d7fb6bc3…`**. That is expected and correct: the pinned evidence — the P
decision, the golden corpora, the falsification records — describes the surface it was taken on,
and remains valid for it. Anything that wants to *measure* again must re-pin.
