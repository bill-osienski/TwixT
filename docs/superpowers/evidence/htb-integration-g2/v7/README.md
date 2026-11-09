# G2 V7 — golden comparison under `protocol-hard-pinned`: MISMATCH, value numerics only (2026-11-09)

## What ran (once)

- **Capture,** in a clean linked worktree at `efc3238a`:
  `node tests/mcts_golden/capture.mjs capture runs/mcts_golden_protocol_hard_pinned --stage protocol-hard-pinned`.
  - Exit 0, 92 of 92 artifacts in 92 distinct processes, `corpus VALID`, 0 bytes on stderr, 59 s.
  - The artifacts are in `artifacts/`, corpus fingerprint `10f383f2…`.
  - Environment: Node v24.22.0, onnxruntime-node 1.23.2, macOS 27.2 (26B123), Apple M5 Max.
- **Comparison:** `node tests/mcts_golden/compare_stage.mjs <dir> --stage protocol-hard-pinned` (`compare_stage.txt`).
  - Frozen `compare.mjs` hard-codes the `lazy` subject and refuses any other stage (`LAZY_CORPUS_INVALID`, 3 stage
    fields per artifact). It was not edited.
  - `compare_stage.mjs` (`e144e7ce`) reuses its exported comparison, fingerprint and validation unchanged, with a
    named subject stage. It was committed before this comparison ran.
  - Controls, run first: it reproduces August's **EXACT MATCH** for the committed lazy corpus under `--stage lazy`,
    and refuses that corpus under the wrong stage.
  - Standard: the frozen eager corpus (`841df60`, fingerprint `9e3a9037…`), captured 2026-08-16/17 with Node v26.7.0
    and onnxruntime-node 1.23.2.

## Result: MISMATCH (exit 1), 12,326 compared-field differences in 71 of 92 cases

| field | differences |
|---|---:|
| `visit_counts`, values and order | **0** (all 92 cases identical) |
| `selected_move` | **0** (all 92 cases identical) |
| `root_value` | 71 cases; abs difference 3.3e-9 to 1.43e-6, median 1.98e-7 |
| `progress.valueEstimate` | 12,255 entries; about 1e-7 to 1e-6, with one exception (below) |

- **By model:**
  - baseline 1d64027: 60 of 80 cases differ;
  - candidate c34b7ff: 10 of 10;
  - abort cases: 1 of 2.
- **The one larger difference:** 3.3e-4, at `G_P11_baseline_s800`, progress entry 632.
  - That one entry differs by 3.3e-4; the entries on either side differ by 2.35e-7, and step 633 agrees again.
  - The final visit counts are identical.
  - This is two simulations in a different order. A near-tie in PUCT selection flips under the about 1e-7 value
    differences, and the end state is the same.

## Reading, and what is not established

- **What differs:** every difference is a value-numerics difference of about 1e-7, which is network output. No
  search decision differs in the end state (visit counts, selected moves).
- **What did not change:**
  - `server/mcts.js`, `server/gameLogic.js` and `server/inference.js` are byte-identical to the stages the standard
    describes;
  - both models are byte-identical (pinned and hashed);
  - onnxruntime-node is the same version.
- **Recorded environment differences from the standard's capture:**
  - Node v24.22.0 here, v26.7.0 then;
  - the operating system has since updated to macOS 27.2.
- **Not established:** whether the environment is the cause.
  - **The discriminator:** recapture at the lazy stage's own commit (`85894b9`) in today's environment, and judge it
    with the unedited `compare.mjs`.
  - If it shows the same pattern, the integration is not involved.

Under compare.mjs's own frozen interpretation, a mismatch is preserved and stops the programme. G2 stopped here.

## Correction (2026-11-09, the user's review)

"Two simulations in a different order" (the `G_P11_baseline_s800` entry 632 transient) is a **plausible explanation,
not something the identical final visit counts prove**. No instrumentation was added to establish it.
