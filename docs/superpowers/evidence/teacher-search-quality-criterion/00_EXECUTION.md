# Teacher search-quality criterion — scan evidence (read-only, 2026-10-10)

These are the scripts and outputs behind §4 of
`docs/superpowers/2026-10-10-teacher-search-quality-criterion-memo.md`. They were preserved after the fact
from the session scratchpad, and **nothing was rerun** to produce this folder.

No model or teacher was loaded, nothing was written to the repository during the scans, and every gate was
closed.

## Execution

* **Where it ran:** from the repository root, at `ca9cb1eb`, with a clean tree. Nothing under `scripts/`
  changed between `ca9cb1eb` and this commit (`git diff --quiet ca9cb1eb HEAD -- scripts`).
* **How:** `PYTHONPATH=. .venv/bin/python <script>`, using Python 3.14.7 and numpy 2.4.1 on macOS 26.6.2
  arm64.
* **Inputs:** the 8 stage-1 training games, families 0–3, under
  `evidence/external-engine-learning-stage1/games/`. Their sha256 are in `05_input_hashes.txt`. Each
  script asserts `split == "train"`. Families 4–5 are never opened.
* **Code:** the repo-side import closure, with sha256, is in `04_source_closure.txt`. It was hashed at
  this commit, which equals `ca9cb1eb` for `scripts/`.

| # | script (sha256 prefix) | started (UTC) | duration | exit | how its output was kept |
|---|---|---|---|---|---|
| 01 | `01_one_ply_scan.py` (`f1b6466b…`) | 2026-10-10 22:22:44 | 84.0 s | 0 | stdout and stderr went to the terminal through `\| tail -60`; the full 29 lines were recovered verbatim from the session transcript (`91_transcript_extract.json`) |
| 02 | `02_bounded_proof_scan.py` (`56de9855…`) | 2026-10-10 22:25:51 | 5,359 s | 0 | redirected to a file, copied unchanged |
| 03 | `03_far_move_probe.py` (`c96c66f5…`) | 2026-10-10 23:57:28 | 1,860 s | 0 | redirected to a file, copied unchanged |
| 90 | `90_INVALID_null_move_shortcut.py` (`7160b0f5…`) | 2026-10-10 23:56:34 | killed after about 54 s | 144 | **none kept**: its output file was overwritten by run 03 |

An attempt of 01 at 22:22:41 failed on import, before reading anything, because `PYTHONPATH` was missing.
It was rerun as shown.

A one-off timing probe measured `classify` at about 4.8 ms per call, over 100 calls on one f0 position.
It is recorded in `91_transcript_extract.json`.

**Identity.** Each preserved script was checked byte-for-byte against the heredoc that wrote it in the
transcript, so these are the files that ran. Script 03 overwrote script 90 at the same scratch path; 90's
source here is recovered from its heredoc.

## 90 is INVALID: diagnostics only

Its premise is false under our rules: "an own peg never removes the mover's own defence". Links are added
automatically and own crossings are forbidden (`TwixtState._find_new_bridges_on_new_state`), so a
player's peg can pre-empt that player's own later link. That can:

* remove an own immediate win; and
* cancel a defensive link that a saving move would otherwise have made.

So its claim "T0, R0 empty ⇒ no move is LOSS4" is unsound. It was stopped and replaced by 03, and none of
its results were used.

## What each output does and does not show

* **01:** the M1 class of every decision, plus `cross_check` against the brute-force scan at every decision whose ply index is a
  multiple of 4 (66 decisions, all agreeing).
* **02:** WIN3 over all legal moves and LOSS4 of the chosen move, both exact.
  * For a chosen LOSS4 move, the script searches every alternative for one that is not LOSS4.
  * `alt` is `None` only when every alternative was LOSS4.
* **03:** LOSS4 tested on only the 3 legal moves farthest from every peg.
  * A hit confirms that a losing move existed.
  * **`no_hit` is not a proof that no losing move exists.** 03 gives a lower bound on positions where a
    mistake was possible, never an upper bound.
