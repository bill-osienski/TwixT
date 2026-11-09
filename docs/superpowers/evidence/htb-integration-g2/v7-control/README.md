# Current-environment equivalence control for V7 (2026-11-09)

This control was declared before capture (ledger decision note of 2026-11-09, commit `09291a98`).

## Capture

- **Command:** `node tests/mcts_golden/capture.mjs capture runs/mcts_golden_control_aad5796 --stage lazy`.
- **Where:** a clean linked worktree at `aad5796` (the original lazy capture commit). Its `compare.mjs`, `worker.mjs`,
  `server/mcts.js`, `server/gameLogic.js` and `server/inference.js` are identical to today's, and its model blobs
  are identical.
- **Environment:** today's: Node v24.22.0, today's `node_modules` (onnxruntime-node 1.23.2), macOS 27.2.
- **Time:** 2026-11-09 17:21:51Z to 17:22:48Z.
- **Outcome:** exit 0, 92 of 92 artifacts in 92 distinct processes, `corpus VALID`, 0 bytes on stderr.
- **Artifacts:** `artifacts/`, fingerprint `30888e31…`.

## Comparisons

| | tool | result |
|---|---|---|
| step 2: the control against August's frozen eager standard | unchanged `compare.mjs` | **MISMATCH**, 12,326 differences (exit 1), the same count as V7 (`control_vs_august_eager.txt`) |
| **step 3: the control against V7's integrated capture** (`10f383f2…`, `protocol-hard-pinned`) | `compare_pair.mjs` | **EXACT MATCH**, 92 cases, on `visit_counts` (values and order), `root_value`, `selected_move` and progress (`done`, `total`, `valueEstimate`); `progress_elapsed_ms` excluded (`control_vs_v7_integrated.txt`) |

## Result

- **Old code and integrated code produce identical traces in the same current environment.** V7's differences from
  August's standard are therefore **not caused by the integration**: the pre-integration code reproduces them exactly.
- **Recorded environment differences** from August's captures: Node v24.22.0 (August v26.7.0) and macOS 27.2.
- **Not established:** which component of the environment changes the network values by about 1e-7.
- **What stays unchanged:** V7's MISMATCH against August, and all historical goldens.
