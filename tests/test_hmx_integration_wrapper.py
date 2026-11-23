"""H-MX integration check wrapper: model-free tests (proposal docs/superpowers/2026-11-22-htb-mix-integration-proposal.md,
revision 1, B7).

No model loads and no gate opens in this repository. The committed wrapper must be its derivation, byte for byte. The
real wrapper then runs (bash, git and node) in throwaway repositories: every refusal before the idle gate writes
nothing, launches nothing and leaves the gate closed; with every precondition met it reaches the idle gate, whose fake
interpreter fails, so it still writes and launches nothing. Run with the main checkout's interpreter:

    /Users/bill/projects/TwixT_Game/.venv/bin/python -m pytest -q tests/test_hmx_integration_wrapper.py
"""
import json
import os
import pathlib
import re
import shutil
import subprocess

import pytest

from tests import hmx_integration_wrapper_spec as W

ROOT = pathlib.Path(__file__).resolve().parents[1]


# ─────────────────────────────── the derivation ───────────────────────────────

def test_THE_COMMITTED_WRAPPER_IS_ITS_DERIVATION():
    text = W.wrapper_path().read_text()
    assert text == W.derive()
    assert os.access(W.wrapper_path(), os.X_OK)
    assert subprocess.run(["bash", "-n", str(W.wrapper_path())]).returncode == 0


def test_THE_DERIVATION_CHANGES_ONLY_THE_PLANNED_LINES():
    body = W.wrapper_path().read_text().split("set -u\n", 1)[1]
    for line in (f"OUT_REL={W.OUT_REL}", f"RUN_REL={W.RUN_REL}", f"GATE_REL={W.GATE_REL}", f"GATE={W.GATE}",
                 f"DRIVER={W.DRIVER}", "STAGE=check", "TERM_S=14390", "KILL_S=14400", "GRACE_S=10",
                 f"MATCH_RECORD_SHA256={W.MATCH_RECORD_SHA256}", "RECORD_KIND=hmx-integration-check-record/1"):
        assert f"\n{line}\n" in f"\n{body}", line
    assert ('env -u MODEL_MANIFEST -u HARD_MODEL_MANIFEST -u HARD_MODE -u NODE_OPTIONS -u HMX_SEED_QUEUE "$NODE" "$DRIVER" '
            'run "$STAGE" --ref-root "$REF_ROOT" --revert-root "$REVERT_ROOT" --kill-s "$KILL_S"') in body
    for gone in ("ANALYSER", "EXPORT_REL", "CHALLENGER_DIR", "TIMING_REL", "htb-mix-match-smoke", "--challenger-dir"):
        assert gone not in body, gone
    # Everything else is the parent's: aligned line by line, every changed block's parent side is one of the planned
    # substitutions (12 settings, the record kind, the analyser check and the export-record block (5), a comment, the
    # smoke and timing block (12), the launch line), 32 parent lines in all.
    import difflib

    parent = (ROOT / W.PARENT).read_text().split("set -u\n", 1)[1].splitlines()
    child = body.splitlines()
    planned = ("OUT_REL=", "RUN_REL=", "PLAN_REL=", "ANALYSER", "EXPORT_REL", "GATE_REL=", "GATE=", "DRIVER=", "STAGE=",
               "TERM_S=", "KILL_S=", "RECORD_KIND=", '[ "$(shasum -a 256 "$ANALYSER"', "# THE EXPORT RECORD", "CHALLENGER_DIR=", '[ -d "$CHALLENGER_DIR" ]',
               "git ls-files --error-unmatch \"$EXPORT_REL\"", "# THE PINNED RUNTIMES", "# THE SMOKE EVIDENCE", "for f in ",
               "         docs/superpowers/evidence/htb-mix-match-smoke-run/", "  git ls-files --error-unmatch \"$f\"", "done",
               '[ "$(git ls-files -- docs/superpowers/evidence/htb-mix-match-smoke/games', "# THE LAUNCH ESTIMATE",
               "TIMING_REL=", 'git ls-files --error-unmatch "$TIMING_REL"', "RECOMPUTED=", '[ "$RECOMPUTED"',
               "\"$NODE\" -e 'const t = JSON.parse", '/usr/bin/perl -e "$SUPERVISOR"')
    changed = []
    for op, i1, i2, _j1, _j2 in difflib.SequenceMatcher(None, parent, child, autojunk=False).get_opcodes():
        if op != "equal":
            changed += parent[i1:i2]
    assert len(changed) == 32, len(changed)
    for ln in changed:
        assert ln.startswith(planned), ln
    for must in ("idle_gate wait", "background_compute check", "server/test_ort_shutdown.js", 'the card pins 1.23.2"',
                 "SUPERVISOR=", "outer.json", "GATE NOT RESTORED", "HEAD is not the one-line opening of $GATE",
                 "pgrep -x UnrealEditor", "set -C"):
        assert must in body, must
    assert body.index("# THE IDLE GATE") < body.index("started=true")


def test_THE_GATE_IT_OPENS_IS_THE_ONE_CLOSED_JS_GATE():
    text = (ROOT / W.GATE_REL).read_text()
    assert re.findall(r"^export const (\w+) = (\w+);$", text, re.M) == [(W.GATE, "false")]


# ─────────────────────────────── the real wrapper, in throwaway repositories ───────────────────────────────

GIT_ID = ["-c", "user.name=t", "-c", "user.email=t@t"]


def _git(cwd, *args):
    subprocess.run(["git", *GIT_ID, *args], cwd=cwd, check=True, capture_output=True)


def _node_modules(root, version="1.23.2", link_to=None):
    if link_to is not None:
        _node_modules(link_to, version)
        (root / "node_modules").symlink_to(link_to / "node_modules")
        return
    pkg = root / "node_modules/onnxruntime-node/package.json"
    pkg.parent.mkdir(parents=True, exist_ok=True)
    pkg.write_text(json.dumps({"name": "onnxruntime-node", "version": version}))


def _run(tmp_path, case):
    """A throwaway repository holding the real derived wrapper, the closed gate, the proposal, the preload, the real
    match record, a passing ONNX Runtime shutdown test stand-in and a fake interpreter that logs and fails; then, as
    `case` says, the one-line gate opening and the two other trees."""
    w = tmp_path / "w"
    wrapper = w / W.OUT_REL / "01_wrapper.sh"
    wrapper.parent.mkdir(parents=True)
    shutil.copyfile(W.wrapper_path(), wrapper)
    wrapper.chmod(0o755)
    gate = w / W.GATE_REL
    gate.parent.mkdir(parents=True)
    gate.write_text(f"/** gate */\nexport const {W.GATE} = false;\n")
    for rel in ("docs/superpowers/2026-11-22-htb-mix-integration-proposal.md", W.PRELOAD, W.MATCH_RECORD):
        (w / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, w / rel)
    if case == "bad_match_record":
        (w / W.MATCH_RECORD).write_text((w / W.MATCH_RECORD).read_text() + " ")
    (w / "server").mkdir()
    (w / "server/inference.js").write_text("// fixture\n")
    # The wrapper runs `node --test server/test_ort_shutdown.js`; the fixture stands in with a passing test.
    (w / "server/test_ort_shutdown.js").write_text("import { test } from 'node:test';\ntest('stand-in', () => {});\n")
    (w / ".gitignore").write_text("node_modules\n.venv\n")
    log = tmp_path / "py_calls.txt"
    py = w / ".venv/bin/python"
    py.parent.mkdir(parents=True)
    py.write_text(f'#!/bin/sh\necho "$@" >> "{log}"\nexit 1\n')
    py.chmod(0o755)
    _node_modules(w, "1.23.1" if case == "old_ort" else "1.23.2")
    _git(w, "init", "-q")
    _git(w, "add", "-A")
    _git(w, "commit", "-qm", "closed")
    if case != "gate_not_opened":
        gate.write_text(f"/** gate */\nexport const {W.GATE} = true;\n")
        _git(w, "commit", "-qam", "open")
    else:
        (w / "server/inference.js").write_text("// fixture 2\n")
        _git(w, "commit", "-qam", "not a gate opening")
    for suffix in ("-ref", "-revert"):
        t = tmp_path / f"w{suffix}"
        t.mkdir()
        if case == f"no{suffix}":
            continue
        _node_modules(t, link_to=(tmp_path / f"elsewhere{suffix}") if case == f"symlink{suffix}" else None)
    r = subprocess.run(["bash", str(wrapper)], env={**os.environ, "HOME": str(tmp_path / "home")}, capture_output=True,
                       text=True, timeout=120)
    return w, r, log


@pytest.mark.parametrize("case, refusal", [
    ("gate_not_opened", r"HEAD does not change only tests/htb_integration/hmx_integration_authorization\.mjs"),
    ("no-ref", r"\S+/w-ref has no real node_modules"),
    ("symlink-revert", r"\S+/w-revert has no real node_modules"),
    ("old_ort", r"onnxruntime-node 1\.23\.1, the card pins 1\.23\.2"),
    ("bad_match_record", r"docs/superpowers/evidence/htb-mix-match/record\.json is not the committed match record"),
    ("ok", r"the machine did not become idle"),
])
def test_THE_REAL_WRAPPER_REFUSES_CLEANLY_AND_NEVER_LAUNCHES(tmp_path, case, refusal):
    """Each miss refuses before the idle gate: the interpreter is never called, nothing is written, the gate is
    restored. With every precondition met the wrapper reaches the idle gate, whose fake interpreter fails, so it
    refuses there: still nothing written and nothing launched."""
    w, r, log = _run(tmp_path, case)
    assert r.returncode == 1, r.stderr
    assert re.search(f"REFUSED: ({refusal})", r.stderr), r.stderr
    if case == "ok":
        calls = log.read_text()
        assert "scripts.GPU.alphazero.idle_gate wait" in calls
        assert "hmx_integration_check" not in calls and "background_compute" not in calls
    else:
        assert not log.exists()
    assert sorted(p.name for p in (w / W.OUT_REL).iterdir()) == ["01_wrapper.sh"]
    assert not (w / W.RUN_REL).exists()
    assert (w / W.GATE_REL).read_text().endswith(f"export const {W.GATE} = false;\n")
    assert "GATE NOT RESTORED" not in r.stderr
