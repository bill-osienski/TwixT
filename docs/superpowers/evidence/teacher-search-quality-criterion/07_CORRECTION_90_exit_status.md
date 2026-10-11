# Correction to `00_EXECUTION.md`: row 90's exit status (2026-10-10)

This supersedes the "exit" cell of row 90 in `00_EXECUTION.md`'s table. Evidence is create-only, so that
file and `06_CORRECTION_01_exit_status.md` are left unchanged; this note corrects the cell.

* **Row 90, exit:** **144 is the wrapper shell's status. Python's exit status was not captured.**
* **Why it is the shell's:**
  * The run was launched as
    `PYTHONPATH=. .venv/bin/python …/qc_scan3.py > …/qc_scan3.out 2>&1; echo exit $?`.
  * That wrapper's whole recorded output is `[exited with code 144]`, with no `exit N` line. So the
    `echo exit $?` that would have printed Python's status never ran.
  * The run was stopped with `pkill -f qc_scan3.py`. Its pattern also appears in the wrapper's own command
    line.
* **Missing output:** the shortcut's output stays not retained, as `00_EXECUTION.md` says. It is not
  reconstructed here.

Nothing was rerun. The earlier manifests stay as they are:
* `MANIFEST.sha256` covers the original 11 files.
* `MANIFEST-2.sha256` covers those plus 06.
* `MANIFEST-3.sha256` is cumulative and covers every non-manifest file, including this one.
