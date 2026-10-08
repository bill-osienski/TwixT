"""The A8.8 step-6 R1 binding record (card §18 A8.6) -- its checks and controls, SYNTHETIC ONLY.

The binding script is imported by path; it is never RUN here against the real tree. R1's node comparison,
the import closure, the registry's one permitted difference, the guard's verdict and the entry's exit
codes are exercised on synthetic sources, temporary git repositories and stand-ins. No band value is
spelled.
"""
import ast
import importlib.util
import pathlib
import re
import subprocess
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs/superpowers/evidence/2026-10-08-tactical-candidate-r1-binding/01_r1_binding.py"


@pytest.fixture(scope="module")
def M():
    spec = importlib.util.spec_from_file_location("_r1_binding", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)


# ─────────────────────────────── the script, its pins and its allowlist ───────────────────────────────

def test_the_script_and_this_file_SPELL_no_band_literal(M):
    spec = importlib.util.spec_from_file_location("_s", ROOT / "docs/superpowers/evidence/"
                                                  "tactical-candidate-seed-survey/01_seed_survey.py")
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)
    for path in (SCRIPT, pathlib.Path(__file__)):
        assert S.scan(path.read_text()) == {}, path


def test_the_PATHS_are_this_script_its_record_and_these_tests(M):
    assert (M.ROOT / M.SCRIPT).resolve() == SCRIPT and M.ROOT == ROOT
    assert (M.ROOT / M.TESTS).resolve() == pathlib.Path(__file__).resolve()
    assert M.OUT == f"{M.HERE}/02_r1_binding_run.txt"
    assert M.CHANGEABLE_SINCE_PRE_PLAY == (M.CARD, M.SCRIPT, M.TESTS)


def test_the_PINS_are_the_qualifications_R1_and_the_frozen_proof(M):
    pins = next(n for n in ast.parse((ROOT / "scripts/GPU/alphazero/tactical_qualification.py").read_text()).body
                if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "CODE_PINS")
    assert ast.literal_eval(pins.value)["candidate_match_runner.py"] == M.R1_BASE_SHA256
    assert M._sha(M.blob(ROOT, M.BASE_COMMIT, M.R1)) == M.R1_BASE_SHA256
    assert M._sha((ROOT / M.PROOF_SCRIPT).read_bytes()) == M.PROOF_SCRIPT_SHA256
    assert _git(ROOT, "merge-base", "--is-ancestor", M.PRE_PLAY_COMMIT, "HEAD").returncode == 0


def _a8_6():
    card = (ROOT / "docs/superpowers/2026-10-05-tactical-candidate-evaluation-card.md").read_text()
    return card[card.index("### A8.6"):card.index("### A8.7")]


def test_every_name_A8_6_SPELLS_in_its_allowlist_is_allowlisted_and_the_rest_is_ITS_categories(M):
    sec = _a8_6()
    text = sec[sec.index("**The allowlist**"):sec.index("Inside `_run`")]
    spelled = set(re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", text))
    assert spelled <= set(M.ALLOWLIST), spelled - set(M.ALLOWLIST)
    # the names A8.6 gives by CATEGORY only -- listed here so a reviewer sees each mapping
    assert set(M.ALLOWLIST) - spelled == {
        "qualification_binding", "_bound_plan", "QUALIFICATION_COMMIT", "QUALIFICATION_FILES",
        "QUALIFICATION_MODULE_SHA256", "QUALIFICATION_CALLS",                    # the guard and its constants
        "_sigint_held",                                                          # the SIGINT hold
        "_write_timing",                                                         # the timing writer
        "_read_timing", "_read_verification",                                    # item-4 readers
        "_finalized_path", "_finalization_failed_path"}                          # item 5's path helpers


def test_the_NAMED_identical_functions_are_exactly_A8_6s_list(M):
    sec = _a8_6()
    text = sec[sec.index("That includes every constant"):sec.index("**The allowlist**")]
    assert set(re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", text)) == set(M.NAMED_IDENTICAL)
    assert not set(M.NAMED_IDENTICAL) & set(M.ALLOWLIST)


# ─────────────────────────────── 1. R1 per node ───────────────────────────────

BASE = '''"""doc"""
import os
from . import helper as H
X = 1


def keep():
    return 1


def _run(a, b):
    g = production_builder(a, b)
    return play_game(g, a)


def run():
    return 0
'''


def _r1(M, src):
    return M.compare_r1(BASE, src, named=("keep",))


def test_R1_IDENTICAL_and_comment_only_versions_bind(M):
    for src in (BASE, "# c\n\n" + BASE.replace("X = 1", "X = 1  # c\n")):
        rows, calls, faults = _r1(M, src)
        assert faults == [] and {s for _, s, _ in rows} == {"IDENTICAL"}
        assert calls == [("production_builder", 1, 1, True), ("play_game", 1, 1, True)]


@pytest.mark.parametrize("old,new,words", [
    ("return 1", "return 2", ["R1 'keep' CHANGED, outside the A8.6 allowlist"]),
    ("X = 1", "X = 2", ["R1 'X' CHANGED, outside"]),
    ('"""doc"""', '"""DOC"""', ["R1 '<module docstring>' CHANGED, outside"]),
    ("helper as H", "other as H", ["'from . import helper as H' REMOVED", "'from . import other as H' ADDED"]),
    ("X = 1", "from . import other\nX = 1", ["'from . import other' ADDED, outside"]),
    ("def keep():\n    return 1\n", "", ["R1 'keep' REMOVED", "'keep', named by A8.6, is not in both versions"]),
    ("production_builder(a, b)", "production_builder(b, a)",
     ["the production_builder(...) call nodes inside _run are not identical (1 at the base, 1 now)"]),
    ("    return play_game(g, a)", "    play_game(g, a)\n    return play_game(g, a)",
     ["the play_game(...) call nodes inside _run are not identical (1 at the base, 2 now)"]),
    ("return play_game(g, a)", "return g", ["play_game(...) call nodes inside _run are not identical (1 at the base, 0 now)"]),
])
def test_R1_a_difference_OUTSIDE_the_allowlist_is_a_FAULT(M, old, new, words):
    faults = _r1(M, BASE.replace(old, new))[2]
    for w in words:
        assert any(w in f for f in faults), (w, faults)


def test_R1_an_ADDED_non_allowlisted_function_is_a_FAULT(M):
    assert _r1(M, BASE + "\n\ndef extra():\n    pass\n")[2] == ["R1 'extra' ADDED, outside the A8.6 allowlist"]


@pytest.mark.parametrize("src,key,phrase", [
    (BASE.replace("return 0", "return 9"), "run", "run"),
    (BASE + "\n\ndef _read_timing():\n    pass\n", "_read_timing", "a new helper that reads a predecessor's timing.json"),
    (BASE.replace("import os", "import os\nimport json"), "import json", "standard-library imports"),
    (BASE.replace("import os\n", ""), "import os", "standard-library imports"),
])
def test_R1_an_ALLOWLISTED_difference_binds_and_is_LABELLED(M, src, key, phrase):
    rows, _calls, faults = _r1(M, src)
    assert faults == []
    assert any(k == key and s != "IDENTICAL" and why and why.startswith(phrase) for k, s, why in rows), rows


def test_R1_the_CALLS_are_checked_even_though_run_is_allowlisted(M):
    rows, _c, faults = _r1(M, BASE.replace("play_game(g, a)", "play_game(g, b)"))
    assert ("_run", "CHANGED", M.ALLOWLIST["_run"]) in rows and len(faults) == 1


def test_R1_a_call_MISSING_from_both_versions_is_a_FAULT(M):
    src = BASE.replace("return play_game(g, a)", "return g")
    assert any("play_game(...) call nodes" in f for f in M.compare_r1(src, src, named=("keep",))[2])


def test_R1_two_nodes_with_ONE_key_STOP(M):
    with pytest.raises(M.BindingStop, match="share the key 'keep'"):
        _r1(M, BASE + "\n\ndef keep():\n    return 1\n")


def test_calls_in_finds_BARE_and_ATTRIBUTE_calls_nested_anywhere(M):
    fn = ast.parse("def _run():\n    if x:\n        CM.play_game(1)\n    play_game(2)\n    play_gameX(3)\n").body[0]
    assert len(M.calls_in(fn, "play_game")) == 2 and M.calls_in(None, "play_game") == []


@pytest.mark.parametrize("src,want", [("import os", True), ("import os.path", True), ("from __future__ import annotations", True),
                                      ("import numpy", False), ("import os, numpy", False), ("from . import x", False),
                                      ("from scripts.GPU import x", False)])
def test_is_stdlib_import(M, src, want):
    assert M.is_stdlib_import(ast.parse(src).body[0]) is want


# ─────────────────────────────── 2. the closure and the registry ───────────────────────────────

REG = ("ACCOUNTED_SEED_INTERVALS = (\n    (100, 132),\n)\nEXPOSED_SEED_INTERVALS = ()\n"
       "RETIRED_SEED_INTERVALS = ()\nTEST_ONLY_SEED_INTERVALS = ()\nOTHER = 1\n")
REG_GOOD = REG.replace("    (100, 132),\n", "    (100, 132),\n    # pilot\n    (200, 232),\n")
TREE = {"pkg/__init__.py": "", "pkg/r1.py": "from . import a\nimport json\n\n\ndef f():\n    from . import lazy\n",
        "pkg/a.py": "from .sub import b\nfrom pkg.sub import c\nfrom . import reg\n",
        "pkg/sub/__init__.py": "", "pkg/sub/b.py": "from .. import deep\n", "pkg/sub/c.py": "C = 1\n",
        "pkg/deep.py": "D = 1\n", "pkg/lazy.py": "Y = 1\n", "pkg/unreached.py": "Z = 1\n", "pkg/reg.py": REG}


@pytest.fixture
def repo(tmp_path):
    for rel, text in TREE.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)
    for args in (["init", "-q"], ["add", "-A"], ["-c", "user.name=t", "-c", "user.email=t@t", "-c", "core.hooksPath=/dev/null",
                                                  "-c", "commit.gpgsign=false", "commit", "-q", "-m", "base"]):
        assert _git(tmp_path, *args).returncode == 0
    return tmp_path, _git(tmp_path, "rev-parse", "HEAD").stdout.strip()


def _edit(root, rel, text):
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_text(text)


def test_the_CLOSURE_follows_function_imports_relative_absolute_parent_and_package_inits(M, repo):
    root, head = repo
    files, ext = M.closure(root / "pkg/r1.py", root)
    assert sorted(str(f.relative_to(root)) for f in files) == sorted(set(TREE) - {"pkg/unreached.py"})
    assert ext == {"json"}
    rows, faults = M.compare_closure(files, root, head, (200, 232), r1="pkg/r1.py", registry="pkg/reg.py")
    assert faults == [] and {r[3] for r in rows} == {"IDENTICAL", "R1 ITSELF -- item 1"}
    assert all(r[1] == r[2] for r in rows)                                   # both hashes recorded


@pytest.mark.parametrize("rel", ["pkg/lazy.py", "pkg/sub/b.py", "pkg/deep.py", "pkg/sub/c.py", "pkg/sub/__init__.py",
                                 "pkg/__init__.py"])
def test_a_REACHED_module_that_differs_is_a_FAULT(M, repo, rel):
    root, head = repo
    _edit(root, rel, (root / rel).read_text() + "# changed\nQ = 1\n")
    files, _ = M.closure(root / "pkg/r1.py", root)
    _rows, faults = M.compare_closure(files, root, head, (200, 232), r1="pkg/r1.py", registry="pkg/reg.py")
    assert faults == [f"{rel}: DIFFERS"]


def test_an_UNREACHED_module_and_R1_ITSELF_are_not_closure_faults(M, repo):
    root, head = repo
    _edit(root, "pkg/unreached.py", "Z = 2\n")
    _edit(root, "pkg/r1.py", TREE["pkg/r1.py"] + "R = 1\n")
    files, _ = M.closure(root / "pkg/r1.py", root)
    assert M.compare_closure(files, root, head, (200, 232), r1="pkg/r1.py", registry="pkg/reg.py")[1] == []


def test_a_reached_module_with_NO_BLOB_at_the_base_is_a_FAULT(M, repo):
    root, head = repo
    _edit(root, "pkg/r1.py", TREE["pkg/r1.py"] + "from . import new\n")
    _edit(root, "pkg/new.py", "N = 1\n")
    files, _ = M.closure(root / "pkg/r1.py", root)
    assert M.compare_closure(files, root, head, (200, 232), r1="pkg/r1.py",
                             registry="pkg/reg.py")[1] == ["pkg/new.py: NO BLOB AT THE BASE"]


@pytest.mark.parametrize("rel,text,why", [
    ("pkg/a.py", TREE["pkg/a.py"] + "import importlib\n", "it imports importlib"),
    ("pkg/a.py", TREE["pkg/a.py"] + "import runpy\n", "it imports runpy"),
    ("pkg/sub/c.py", "C = __import__('os')\n", r"a dynamic __import__\(\)"),
    ("pkg/sub/c.py", "exec('x = 1')\n", r"a dynamic exec\(\)"),
    ("pkg/sub/c.py", "m = something.import_module('x')\n", r"a dynamic \.import_module\(\)"),
    ("pkg/a.py", TREE["pkg/a.py"] + "from . import gone\n", "names no repository module"),
    ("pkg/a.py", TREE["pkg/a.py"] + "from .gone import x\n", "names no repository module"),
    ("pkg/a.py", TREE["pkg/a.py"] + "import pkg.gone\n", "names no repository module"),
    ("pkg/json.py", "J = 1\n", "could resolve to a file beside it"),
])
def test_an_import_the_closure_CANNOT_resolve_STOPS(M, repo, rel, text, why):
    root, _head = repo
    _edit(root, rel, text)
    with pytest.raises(M.BindingStop, match=why):
        M.closure(root / "pkg/r1.py", root)


@pytest.mark.parametrize("text,ok", [
    (REG_GOOD, True),
    ("# a comment\n" + REG_GOOD, True),
    ("# a comment\n" + REG, False),
    (REG.replace("(\n    (100, 132),", "(\n    (200, 232),\n    (100, 132),"), False),
    (REG_GOOD.replace("(200, 232)", "(200, 233)"), False),
    (REG_GOOD.replace("(200, 232),\n", "(200, 232),\n    (300, 332),\n"), False),
    (REG_GOOD.replace("EXPOSED_SEED_INTERVALS = ()", "EXPOSED_SEED_INTERVALS = ((1, 2),)"), False),
    (REG_GOOD.replace("OTHER = 1", "OTHER = 2"), False),
    (REG_GOOD + "NEW = 1\n", False),
])
def test_the_REGISTRY_may_differ_ONLY_by_the_block_appended_last(M, repo, text, ok):
    root, head = repo
    _edit(root, "pkg/reg.py", text)
    files, _ = M.closure(root / "pkg/r1.py", root)
    rows, faults = M.compare_closure(files, root, head, (200, 232), r1="pkg/r1.py", registry="pkg/reg.py")
    assert (faults == []) is ok, faults
    verdict = dict((r[0], r[3]) for r in rows)["pkg/reg.py"]
    assert verdict == "ONLY THE A8.2 TUPLE" if ok else verdict.startswith("DIFFERS: ")


@pytest.mark.parametrize("rows,rf,want", [
    ([("r", "a", "a", "IDENTICAL")], ["gone"], ["registry: gone"]),          # bytes equal the base: block missing
    ([], ["gone"], ["registry: gone"]),                                      # not reached: still checked
    ([("r", "a", "b", "DIFFERS: gone")], ["gone"], []),                      # counted once, by the closure
    ([("r", "a", "b", "ONLY THE A8.2 TUPLE")], [], []),
])
def test_the_REGISTRY_is_checked_even_where_the_closure_did_not_flag_it(M, rows, rf, want):
    assert M.registry_extra_faults(rows, rf, registry="r") == want


@pytest.mark.parametrize("cur,words", [
    (REG + "ACCOUNTED_SEED_INTERVALS = ()\n", "not assigned exactly once"),
    (REG.replace("ACCOUNTED_SEED_INTERVALS = (\n    (100, 132),\n)", "ACCOUNTED_SEED_INTERVALS = tuple(BASE)"),
     "not a literal"),
    (REG_GOOD.replace("EXPOSED_SEED_INTERVALS = ()", "EXPOSED_SEED_INTERVALS = ((1, 2),)"),
     "registry statement EXPOSED_SEED_INTERVALS changed"),
    ("X = 0\n" + REG_GOOD, "added, removed or moved"),
])
def test_registry_faults_NAME_what_moved(M, cur, words):
    assert any(words in f for f in M.registry_faults(REG, cur, (200, 232)))


def test_accounted_of_reads_the_literal_or_nothing(M):
    assert M.accounted_of(REG_GOOD) == ((100, 132), (200, 232))
    assert M.accounted_of("X = 1\n") == () and M.accounted_of(REG + REG) == ()


# ─────────────────────────────── 3. the guard and the pre-play state ───────────────────────────────

class Refused(Exception):
    pass


PAIRS = [{"pair_id": f"p-{k}", "red": 300 + 2 * k, "black": 301 + 2 * k} for k in range(2)]
PLAN = {"pairs": [{"pair_id": p["pair_id"], "seeds": {"red": p["red"], "black": p["black"]}} for p in PAIRS]}
BINDING = {"commit": "c", "corpus_sha256": "x", "call_list_sha256": "y"}


def _refuse(*_a):
    raise Refused("refused here")


def _cm(**over):
    ns = dict(MatchRefused=Refused, qualification_binding=lambda: BINDING, _committed_plan=lambda r, s: (PLAN, "sha"),
              _bound_plan=lambda p: BINDING, out_dir=lambda r, s: pathlib.Path("/nonexistent/r1-binding-test"))
    ns.update(over)
    return types.SimpleNamespace(**ns)


def _guard(M, cm, pairs=PAIRS, block=(300, 304), acc=((300, 304),), gates=()):
    return M.guard_checks(cm, pairs, block, acc, list(gates))


def test_the_GUARD_holds_and_RECORDS_what_it_bound(M):
    rows, faults = _guard(M, _cm())
    assert faults == [] and all(s == "HOLDS" for _, s, _ in rows) and len(rows) == 8
    assert "commit c; corpus x; call list y" in rows[0][2] and "plan sha256 sha" in rows[1][2]


@pytest.mark.parametrize("over,label", [
    (dict(qualification_binding=_refuse), "qualification_binding() -- A8.4 items 1-5: REFUSED"),
    (dict(_committed_plan=_refuse), '_committed_plan("pilot", 0) -- committed, check_plan: REFUSED'),
    (dict(_bound_plan=_refuse), "_bound_plan(plan) -- A8.4 item 6: REFUSED"),
    (dict(out_dir=lambda r, s: pathlib.Path("/")), "the pilot destination does not exist (pre-play): FAILS"),
])
def test_a_REFUSING_guard_or_an_occupied_destination_is_a_FAULT(M, over, label):
    rows, faults = _guard(M, _cm(**over))
    assert label in faults
    assert any(s == "REFUSED" and d == "refused here" for _, s, d in rows) or "FAILS" in label


@pytest.mark.parametrize("kw,label", [
    (dict(pairs=[dict(PAIRS[0]), dict(PAIRS[1], black=399)]), "the plan's pairs are the survey's pairs, ids and seeds: FAILS"),
    (dict(block=(300, 306)), "the survey's pairs are exactly the block's seeds: FAILS"),
    (dict(acc=((300, 302),)), "ACCOUNTED holds the block: FAILS"),
    (dict(gates=[("m", "G")]), "every gate is closed: FAILS"),
])
def test_the_PLAN_the_BLOCK_ACCOUNTED_and_the_GATES_are_checked(M, kw, label):
    assert label in _guard(M, _cm(), **kw)[1]


def test_an_UNEXPECTED_guard_error_STOPS(M):
    with pytest.raises(M.BindingStop, match="RuntimeError: boom"):
        _guard(M, _cm(qualification_binding=lambda: (_ for _ in ()).throw(RuntimeError("boom"))))


@pytest.mark.parametrize("changed,ancestor,n", [([], True, 0), (["docs/superpowers/2026-10-05-tactical-candidate-evaluation-card.md"], True, 0),
                                                (["scripts/GPU/alphazero/e4_screen_reference.py"], True, 1),
                                                (["docs/superpowers/evidence/tactical-candidate-plans/pilot-s0.json"], True, 1),
                                                ([], False, 1)])
def test_the_PRE_PLAY_state_permits_only_the_card_the_script_and_its_tests(M, changed, ancestor, n):
    assert len(M.pre_play_faults(changed, ancestor)) == n


def test_the_RUN_controls_all_PASS(M):
    results = M._controls()
    assert len(results) >= 40 and [label for label, ok, _ in results if not ok] == []


# ─────────────────────────────── the entry ───────────────────────────────

def test_the_ENTRY_takes_no_arguments(M, monkeypatch):
    def reached():
        raise AssertionError("the entry read the tree before refusing its arguments")
    monkeypatch.setattr(M, "_preconditions", reached)
    with pytest.raises(M.BindingRefused, match="no arguments"):
        M.main(["x", "--force"])


@pytest.fixture
def entry(M, monkeypatch, tmp_path):
    (tmp_path / M.HERE).mkdir(parents=True)
    monkeypatch.setattr(M, "ROOT", tmp_path)
    monkeypatch.setattr(M, "_LINES", [])
    monkeypatch.setattr(M, "_preconditions", lambda: {"head": "h", "block": (300, 304), "pairs": PAIRS})
    return tmp_path


@pytest.mark.parametrize("faults,controls,rc,words", [
    ([], [("c", True, "")], 0, "BINDING VERDICT: BOUND"),
    (["x"], [("c", True, "")], 1, "BINDING VERDICT: NOT BOUND -- 1 difference(s)"),
    ([], [("c", True, ""), ("d", False, "")], 1, "BINDING VERDICT: NOT BOUND -- 1 control(s) failed"),
    ([], [], 1, "NOT BOUND -- 0 control(s) failed"),
])
def test_the_ENTRY_exit_code_and_record_say_ONE_thing(M, monkeypatch, entry, faults, controls, rc, words):
    monkeypatch.setattr(M, "_checks", lambda ctx: list(faults))
    monkeypatch.setattr(M, "_controls", lambda: list(controls))
    assert M.main(["x"]) == rc
    assert words in (entry / M.OUT).read_text()


@pytest.mark.parametrize("exc", [lambda M: M.BindingStop("cannot"), lambda M: OSError("disk")])
def test_a_STOP_or_an_UNFORESEEN_error_is_RECORDED_with_NO_VERDICT(M, monkeypatch, entry, exc):
    def boom(ctx):
        raise exc(M)
    monkeypatch.setattr(M, "_checks", boom)
    assert M.main(["x"]) == 2
    text = (entry / M.OUT).read_text()
    assert "BINDING STOPPED" in text and "NO VERDICT" in text and "BINDING VERDICT" not in text


def test_the_RECORD_is_CREATE_ONLY(M, monkeypatch, tmp_path):
    monkeypatch.setattr(M, "_LINES", ["a"])
    M._write(tmp_path / "r.txt")
    with pytest.raises(FileExistsError):
        M._write(tmp_path / "r.txt")
    assert (tmp_path / "r.txt").read_text() == "a\n"


def test_PRECONDITIONS_refuse_an_existing_record_a_dirty_tree_and_a_missing_commit(M, monkeypatch, repo):
    root, _head = repo
    monkeypatch.setattr(M, "ROOT", root)
    with pytest.raises(M.BindingRefused, match="not in this repository"):
        M._preconditions()
    _edit(root, "stray.txt", "x")
    with pytest.raises(M.BindingRefused, match="not clean"):
        M._preconditions()
    _edit(root, M.OUT, "x")
    with pytest.raises(M.BindingRefused, match="runs once"):
        M._preconditions()


def test_PRECONDITIONS_refuse_a_base_R1_that_is_not_the_pin(M, monkeypatch, repo):
    root, _head = repo
    _edit(root, M.R1, "x = 1\n")                                         # present at the base, other bytes
    for args in (["add", "-A"], ["-c", "user.name=t", "-c", "user.email=t@t", "-c", "core.hooksPath=/dev/null",
                                 "-c", "commit.gpgsign=false", "commit", "-q", "-m", "r1"]):
        assert _git(root, *args).returncode == 0
    head = _git(root, "rev-parse", "HEAD").stdout.strip()
    monkeypatch.setattr(M, "ROOT", root)
    monkeypatch.setattr(M, "BASE_COMMIT", head)
    monkeypatch.setattr(M, "PRE_PLAY_COMMIT", head)
    with pytest.raises(M.BindingRefused, match="not the qualification's pinned R1"):
        M._preconditions()


# ─────────────────────────────── the three items TOGETHER, on a synthetic repository ───────────────────────────────

@pytest.fixture
def world(M, monkeypatch, tmp_path):
    """A repository whose base commit holds R1 (every function A8.6 names, and _run's two calls), a module
    it imports and the registry; the registration then appends the block. Runtime stand-ins for item 3."""
    r1 = ("from . import dep\n\n\n" + "\n\n\n".join(f"def {n}():\n    return 0" for n in M.NAMED_IDENTICAL) +
          "\n\n\ndef _run(a):\n    g = production_builder(a)\n    return play_game(g)\n")
    for rel, text in {"pkg/__init__.py": "", "pkg/r1.py": r1, "pkg/dep.py": "from . import reg\nD = 1\n", "pkg/reg.py": REG}.items():
        _edit(tmp_path, rel, text)
    for args in (["init", "-q"], ["add", "-A"], ["-c", "user.name=t", "-c", "user.email=t@t", "-c", "core.hooksPath=/dev/null",
                                                  "-c", "commit.gpgsign=false", "commit", "-q", "-m", "base"]):
        assert _git(tmp_path, *args).returncode == 0
    _edit(tmp_path, "pkg/reg.py", REG_GOOD)
    pairs = [{"pair_id": f"pilot-{k:02d}", "red": 200 + 2 * k, "black": 201 + 2 * k} for k in range(16)]
    plan = {"pairs": [{"pair_id": p["pair_id"], "seeds": {"red": p["red"], "black": p["black"]}} for p in pairs]}
    state = {"cm": dict(MatchRefused=Refused, qualification_binding=lambda: BINDING, _bound_plan=lambda p: BINDING,
                        _committed_plan=lambda r, s: (plan, "sha"), AUTH=types.SimpleNamespace(CANDIDATE_MATCH_AUTHORIZED=False),
                        out_dir=lambda r, s: tmp_path / "no-such-destination"),
             "open": []}
    for name, value in (("ROOT", tmp_path), ("R1", "pkg/r1.py"), ("REGISTRY", "pkg/reg.py"), ("_LINES", []),
                        ("BASE_COMMIT", _git(tmp_path, "rev-parse", "HEAD").stdout.strip())):
        monkeypatch.setattr(M, name, value)
    monkeypatch.setattr(M, "_runtime", lambda: (types.SimpleNamespace(**state["cm"]), types.SimpleNamespace(
        discover_gates=lambda: {("m", "G"): False}, open_gates=lambda: list(state["open"]))))
    ctx = {"changed": [], "ancestor": True, "base_r1": r1, "cur_r1": r1, "block": (200, 232), "pairs": pairs}
    return tmp_path, ctx, state


def test_CHECKS_on_a_sound_synthetic_world_find_NOTHING(M, world):
    _root, ctx, _state = world
    assert M._checks(ctx) == []
    text = "\n".join(M._LINES)
    assert sum(line == "  HOLDS" for line in M._LINES) == 4 and "FAILS" not in text
    assert "in the closure: True): ONLY THE A8.2 TUPLE (200, 232)" in text


def test_CHECKS_carry_an_ITEM_1_fault(M, world):
    root, ctx, _state = world
    ctx["cur_r1"] = ctx["cur_r1"].replace("def _t():\n    return 0", "def _t():\n    return 1")
    assert M._checks(ctx) == ["R1 '_t' CHANGED, outside the A8.6 allowlist"]


@pytest.mark.parametrize("rel,text,fault", [
    ("pkg/dep.py", "from . import reg\nD = 2\n", ["pkg/dep.py: DIFFERS"]),
    ("pkg/__init__.py", "# x\n", ["pkg/__init__.py: DIFFERS"]),
    # byte-identical to the base: the closure sees IDENTICAL, the registry check still fails -- and so does item 3
    ("pkg/reg.py", REG, ["registry: ACCOUNTED is not the base tuple plus exactly the pilot block, appended last",
                         "ACCOUNTED holds the block: FAILS"]),
    ("pkg/reg.py", REG_GOOD.replace("OTHER = 1", "OTHER = 2"), ["pkg/reg.py: DIFFERS: registry statement OTHER changed"]),
])
def test_CHECKS_carry_an_ITEM_2_fault(M, world, rel, text, fault):
    root, ctx, _state = world
    _edit(root, rel, text)
    assert M._checks(ctx) == fault


@pytest.mark.parametrize("change,fault", [
    (lambda s: s["cm"].update(qualification_binding=_refuse), "qualification_binding() -- A8.4 items 1-5: REFUSED"),
    (lambda s: s["cm"].update(_bound_plan=_refuse), "_bound_plan(plan) -- A8.4 item 6: REFUSED"),
    (lambda s: s.update(open=[("m", "G")]), "every gate is closed: FAILS"),
    (lambda s: s["cm"].update(out_dir=lambda r, q: pathlib.Path("/")), "the pilot destination does not exist (pre-play): FAILS"),
])
def test_CHECKS_carry_an_ITEM_3_fault(M, world, change, fault):
    _root, ctx, state = world
    change(state)
    assert M._checks(ctx) == [fault]


def test_CHECKS_carry_a_PRE_PLAY_fault(M, world):
    _root, ctx, _state = world
    ctx["changed"] = ["pkg/reg.py"]
    assert M._checks(ctx) == ["pkg/reg.py changed since the pre-play commit"]


def test_CHECKS_STOP_where_the_closure_cannot_be_established(M, world):
    root, ctx, _state = world
    _edit(root, "pkg/dep.py", "import importlib\n")
    with pytest.raises(M.BindingStop, match="it imports importlib"):
        M._checks(ctx)
