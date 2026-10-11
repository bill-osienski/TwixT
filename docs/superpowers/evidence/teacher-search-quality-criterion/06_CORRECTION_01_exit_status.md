# Correction to `00_EXECUTION.md`: scan 01's exit status (2026-10-10)

This supersedes the "exit" cell of row 01 in `00_EXECUTION.md`'s table. Evidence is create-only, so that
file is left unchanged and this note corrects it.

* **Row 01, exit:** **not captured; completion inferred from final output.**
* **Why it wasn't captured:** the command was
  `PYTHONPATH=. timeout 590 .venv/bin/python …/qc_scan.py 2>&1 | tail -60; git status --short | head`.
  The status the session reported belongs to that pipeline and the command after it, not to Python. No
  separate `$?` was recorded.
* **What the completion rests on:** the output's final line, `cross-checks 66 all agree; seconds 84.0`.
  The script prints that line only after every decision has been processed and every cross-check
  assertion has passed.

Nothing was rerun. The other rows are unchanged by this note. `MANIFEST.sha256` still covers the original
11 files, and `MANIFEST-2.sha256` adds this one.
