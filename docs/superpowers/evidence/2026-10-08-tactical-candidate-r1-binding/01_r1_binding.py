"""TACTICAL CANDIDATE, A8.8 STEP 6 -- THE R1 BINDING RECORD. READS CODE ONLY.

Card `docs/superpowers/2026-10-05-tactical-candidate-evaluation-card.md` §18 A8.6 (its allowlist
clarified at e8c80602: `_timing_path`, `verification_path`). Runs ONCE, on the clean committed pre-play
state, and writes ONE create-only record beside this file: 02_r1_binding_run.txt. No model, no game, no
gate, no decision.

  1. R1 (`candidate_match_runner.py`) against its blob at d815461e (sha256 a53e7ac8..., the qualification's
     CODE_PINS), per top-level node (`ast.dump`, attributes excluded): every node outside A8.6's
     allowlist is identical, including the functions A8.6 names; inside `_run`, the
     production_builder(...) and play_game(...) call nodes are identical.
  2. THE IMPORT CLOSURE: every repository module R1 reaches by import -- an import anywhere in a module,
     transitively, with the package __init__ files the import runs -- other than R1, is byte-identical to
     its blob at d815461e. The ONE permitted difference: e4_screen_reference.py, by the A8.2 tuple only
     (by parsing: ACCOUNTED is the d815461e tuple plus exactly the pilot block, appended last; every other
     statement, the other three registries among them, identical). A dynamic import, or an import this
     file cannot resolve, STOPS: the closure would be unknown.
  3. THE GUARD (A8.4) on the real committed tree, pure: R1's own qualification_binding(),
     _committed_plan("pilot", 0) and _bound_plan(); the plan's pairs are the survey's, its block in
     ACCOUNTED; the pilot destination does not exist; every gate is closed.
  And THE PRE-PLAY STATE: since aafdbc77 (step 5's post-proof) only the card, this file and its tests
  changed -- production code, registry, plan and historical evidence are as step 5 left them.

Any difference outside the allowlist or beyond the permitted tuple, a refusing guard or a failed control:
NOT BOUND, exit 1 -- the pilot is not authorized. BOUND only if every check and control holds: exit 0.
STOPPED, no verdict: 2. Refused before anything is checked or written: 4. No retry; nothing is adjusted.
The record SUPPORTS the line-by-line review of the allowlisted changes; it does not replace it.

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B <this file>
"""
import ast
import hashlib
import importlib.util
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import types

ROOT = pathlib.Path(__file__).resolve().parents[4]
HERE = "docs/superpowers/evidence/2026-10-08-tactical-candidate-r1-binding"
SCRIPT = f"{HERE}/01_r1_binding.py"
OUT = f"{HERE}/02_r1_binding_run.txt"
TESTS = "tests/test_tactical_r1_binding.py"
CARD = "docs/superpowers/2026-10-05-tactical-candidate-evaluation-card.md"
PKG = "scripts/GPU/alphazero"
R1 = f"{PKG}/candidate_match_runner.py"
REGISTRY = f"{PKG}/e4_screen_reference.py"
BASE_COMMIT = "d815461eb8daa7f462dc8c492316717a0b3045b4"          # the accepted qualification (A8.4)
R1_BASE_SHA256 = "a53e7ac857f045a281af85d7dd268445e2419e1c71fd2a0458092a5c2c7e21b6"
PRE_PLAY_COMMIT = "aafdbc7726344a9c7c2dcf121b5ed9322f465dbf"      # step 5's post-proof
CHANGEABLE_SINCE_PRE_PLAY = (CARD, SCRIPT, TESTS)
PROOF_SCRIPT = "docs/superpowers/evidence/tactical-candidate-seed-registration/01_collision_proof.py"
PROOF_SCRIPT_SHA256 = "dd192011938ca5c6674092b243681174e669e1e96b22b8d13da2a264b340764d"

#: A8.6's allowlist -- may change, or be added -- each name with the card's own words. Nothing else.
ALLOWLIST = {
    "qualification_binding": "the guard (A8.4)",
    "_bound_plan": "the guard (A8.4 item 6: the plan is bound)",
    "QUALIFICATION_COMMIT": "the guard's constants",
    "QUALIFICATION_FILES": "the guard's constants",
    "QUALIFICATION_MODULE_SHA256": "the guard's constants",
    "QUALIFICATION_CALLS": "the guard's constants",
    "_create": "_create",
    "_sigint_held": "the SIGINT hold",
    "_record_parent_failure": "_record_parent_failure",
    "_T_START": "_T_START (A8.5 item 3)",
    "_write_timing": "the timing writer (A8.5 item 3)",
    "_validate_segment": "_validate_segment (A8.5 item 4)",
    "_segment_limit": "_segment_limit (A8.5 item 4)",
    "_read_timing": "a new helper that reads a predecessor's timing.json (A8.5 item 4)",
    "_read_verification": "a new helper that reads a predecessor's verification record (A8.5 item 4)",
    "_nonneg": "_nonneg",
    "_finalize": "_finalize (A8.5 item 5)",
    "_write_finalized": "_write_finalized (A8.5 item 5)",
    "_read_finalization": "_read_finalization (A8.5 item 5)",
    "_finalized_path": "their path helpers (A8.5 item 5)",
    "_finalization_failed_path": "their path helpers (A8.5 item 5)",
    "_timing_path": "_timing_path (clarified at e8c80602)",
    "verification_path": "verification_path (clarified at e8c80602)",
    "_claim": "_claim",
    "run": "run",
    "run_claimed": "run_claimed",
    "_run": "_run (its two call nodes checked apart)",
    "main": "main",
}
STDLIB_PHRASE = "standard-library imports"
#: named by A8.6 as identical -- present in both versions, never allowlisted
NAMED_IDENTICAL = ("_t", "_sha", "check_plan", "production_builder", "_alarm", "_compact", "game_digest",
                   "play_game", "first_override", "check_pair", "_failure", "_file_sha", "_fsync_dir",
                   "_Partial", "_game_file", "_write", "_write_all", "_save_game", "_unconsumed",
                   "_accept_claim", "_open_game", "_watch_games", "_inventory", "_settle", "_gate", "_role",
                   "out_dir", "plan_path", "_committed_plan", "_child_argv")
CALLS_IN_RUN = ("production_builder", "play_game")
STDLIB = frozenset(sys.stdlib_module_names)
DYNAMIC_MODULES = frozenset({"importlib", "runpy", "imp", "pkgutil", "zipimport"})
DYNAMIC_CALLS = frozenset({"__import__", "exec", "eval"})
DYNAMIC_ATTRS = frozenset({"import_module", "spec_from_file_location", "exec_module", "load_module",
                           "run_path", "run_module"})
assert not set(NAMED_IDENTICAL) & set(ALLOWLIST), "a function A8.6 names as identical is allowlisted"


class BindingRefused(Exception):
    """A precondition does not hold: nothing is checked or written."""


class BindingStop(Exception):
    """A check cannot be carried out: the record says STOPPED, with no verdict."""


_LINES = []


def say(line=""):
    _LINES.append(line)
    print(line, flush=True)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)


def blob(root, commit, rel):
    """rel's bytes at commit, or None where it has none."""
    r = subprocess.run(["git", "-C", str(root), "cat-file", "blob", f"{commit}:{rel}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


# ─────────────────────────────── 1. R1, node by node ───────────────────────────────

def _key(i, n):
    if i == 0 and isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str):
        return "<module docstring>"
    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return n.name
    if isinstance(n, ast.Assign):
        return ", ".join(ast.unparse(t) for t in n.targets)
    if isinstance(n, ast.AnnAssign):
        return ast.unparse(n.target)
    if isinstance(n, (ast.Import, ast.ImportFrom)):
        return ast.unparse(n)
    return f"<{type(n).__name__}> {ast.unparse(n).splitlines()[0]}"


def top_nodes(src, what):
    """{key: node} of a module's top level; two nodes with one key STOP (the comparison is ambiguous)."""
    out = {}
    for i, n in enumerate(ast.parse(src).body):
        k = _key(i, n)
        if k in out:
            raise BindingStop(f"{what}: two top-level nodes share the key {k!r}")
        out[k] = n
    return out


def is_stdlib_import(n):
    if isinstance(n, ast.Import):
        return all(a.name.split(".")[0] in STDLIB for a in n.names)
    return isinstance(n, ast.ImportFrom) and n.level == 0 and n.module.split(".")[0] in STDLIB


def calls_in(fn, name):
    """Every call of `name` (bare or as an attribute) inside fn, as sorted ast.dumps."""
    if fn is None:
        return []
    return sorted(ast.dump(c) for c in ast.walk(fn) if isinstance(c, ast.Call) and (
        (isinstance(c.func, ast.Name) and c.func.id == name) or
        (isinstance(c.func, ast.Attribute) and c.func.attr == name)))


def compare_r1(base_src, cur_src, allowlist=ALLOWLIST, named=NAMED_IDENTICAL, calls=CALLS_IN_RUN):
    """Item 1: (rows [(key, state, phrase or None)], call rows, faults)."""
    old, new = top_nodes(base_src, "R1 at the base"), top_nodes(cur_src, "R1 now")
    rows, faults = [], []
    for k in list(old) + [k for k in new if k not in old]:
        o, n = old.get(k), new.get(k)
        if o is not None and n is not None and ast.dump(o) == ast.dump(n):
            rows.append((k, "IDENTICAL", None))
            continue
        state = "CHANGED" if o is not None and n is not None else ("ADDED" if o is None else "REMOVED")
        node = n if n is not None else o
        why = STDLIB_PHRASE if is_stdlib_import(node) else allowlist.get(k)
        rows.append((k, state, why))
        if why is None:
            faults.append(f"R1 {k!r} {state}, outside the A8.6 allowlist")
    for name in named:
        if name not in old or name not in new:
            faults.append(f"R1 {name!r}, named by A8.6, is not in both versions")
    call_rows = []
    for name in calls:
        a, b = calls_in(old.get("_run"), name), calls_in(new.get("_run"), name)
        same = bool(a) and a == b
        call_rows.append((name, len(a), len(b), same))
        if not same:
            faults.append(f"the {name}(...) call nodes inside _run are not identical "
                          f"({len(a)} at the base, {len(b)} now)")
    return rows, call_rows, faults


# ─────────────────────────────── 2. the import closure ───────────────────────────────

def _module_file(path):
    """`path` names a module without its suffix: its .py file, or its package's __init__.py; else None."""
    f = path.parent / (path.name + ".py")
    if f.is_file():
        return f
    if (path / "__init__.py").is_file():
        return path / "__init__.py"
    return None


def _inits(f, root):
    """The package __init__ files that importing f runs (those below the root)."""
    out, d = [], f.parent
    while d != root and root in d.parents:
        i = d / "__init__.py"
        if i.is_file() and i != f:
            out.append(i)
        d = d.parent
    return out


def imports_of(f, root):
    """(repository files f imports, external top-level names): EVERY import anywhere in f -- a function's
    too. STOPS on a dynamic import and on any import it cannot resolve."""
    rel = f.relative_to(root)
    repo, ext = [], set()

    def stop(n, why):
        raise BindingStop(f"{rel}:{n.lineno}: {why} -- the closure cannot be established")
    for n in ast.walk(ast.parse(f.read_bytes(), filename=str(rel))):
        if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Name) and n.func.id in DYNAMIC_CALLS:
                stop(n, f"a dynamic {n.func.id}()")
            if isinstance(n.func, ast.Attribute) and n.func.attr in DYNAMIC_ATTRS:
                stop(n, f"a dynamic .{n.func.attr}()")
        elif isinstance(n, ast.ImportFrom) and n.level:
            base = f.parent
            for _ in range(n.level - 1):
                base = base.parent
            dots = "." * n.level
            if n.module:
                target = base.joinpath(*n.module.split("."))
                m = _module_file(target)
                if m is None:
                    stop(n, f"'from {dots}{n.module}' names no repository module")
                repo.append(m)
                if m.name == "__init__.py":
                    repo += [s for a in n.names if (s := _module_file(target / a.name)) is not None]
            else:
                for a in n.names:
                    m = _module_file(base / a.name)
                    if m is None:
                        stop(n, f"'from {dots} import {a.name}' names no repository module")
                    repo.append(m)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in n.names] if isinstance(n, ast.Import) else [n.module]
            for name in names:
                parts = name.split(".")
                top = parts[0]
                if top in DYNAMIC_MODULES:
                    stop(n, f"it imports {top}")
                if f.parent != root and ((f.parent / (top + ".py")).exists() or (f.parent / top).is_dir()):
                    stop(n, f"the absolute import {name!r} could resolve to a file beside it")
                if (root / (top + ".py")).is_file() or (root / top).is_dir():
                    m = _module_file(root.joinpath(*parts))
                    if m is None:
                        stop(n, f"{name!r} names no repository module")
                    repo.append(m)
                    if isinstance(n, ast.ImportFrom) and m.name == "__init__.py":
                        repo += [s for a in n.names if (s := _module_file(m.parent / a.name)) is not None]
                else:
                    ext.add(top)
    return repo, ext


def closure(start, root):
    """(every repository file reached from start, sorted; the external top-level names)."""
    seen, todo, ext = set(), [start], set()
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        seen.add(f)
        repo, e = imports_of(f, root)
        ext |= e
        todo += [m for m in repo + _inits(f, root) if m not in seen]
    return sorted(seen), ext


def _assigned(n, name):
    return isinstance(n, ast.Assign) and [ast.unparse(t) for t in n.targets] == [name]


def accounted_of(src):
    """ACCOUNTED_SEED_INTERVALS as parsed, or () where it is not one literal assignment."""
    found = [n for n in ast.parse(src).body if _assigned(n, "ACCOUNTED_SEED_INTERVALS")]
    try:
        return ast.literal_eval(found[0].value) if len(found) == 1 else ()
    except ValueError:
        return ()


def registry_faults(base_src, cur_src, block):
    """A8.2/A8.6: the registry differs from the base ONLY by the block appended last to ACCOUNTED."""
    old, new = ast.parse(base_src).body, ast.parse(cur_src).body
    io = [i for i, n in enumerate(old) if _assigned(n, "ACCOUNTED_SEED_INTERVALS")]
    inew = [i for i, n in enumerate(new) if _assigned(n, "ACCOUNTED_SEED_INTERVALS")]
    if len(io) != 1 or len(inew) != 1:
        return ["ACCOUNTED_SEED_INTERVALS is not assigned exactly once"]
    try:
        vo, vn = ast.literal_eval(old[io[0]].value), ast.literal_eval(new[inew[0]].value)
    except ValueError:
        return ["ACCOUNTED_SEED_INTERVALS is not a literal"]
    faults = []
    if vn != vo + (tuple(block),):
        faults.append("ACCOUNTED is not the base tuple plus exactly the pilot block, appended last")
    if len(old) != len(new) or io != inew:
        return faults + ["a registry statement was added, removed or moved"]
    for i, (a, b) in enumerate(zip(old, new)):
        if i != io[0] and ast.dump(a) != ast.dump(b):
            what = ", ".join(ast.unparse(t) for t in a.targets) if isinstance(a, ast.Assign) else type(a).__name__
            faults.append(f"registry statement {what} changed")
    return faults


def registry_extra_faults(rows, rf, registry=REGISTRY):
    """The registry's own faults that the closure rows have not already counted: it is checked whether or
    not the closure reached it, and even where its bytes equal the base's (the block then missing)."""
    counted = any(r[0] == registry and r[3].startswith("DIFFERS") for r in rows)
    return [] if counted else [f"registry: {x}" for x in rf]


def compare_closure(files, root, base_commit, block, r1=R1, registry=REGISTRY):
    """Item 2: (rows [(rel, sha now, sha at base or None, verdict)], faults). R1 is item 1's."""
    rows, faults = [], []
    for f in files:
        rel = str(f.relative_to(root))
        now, then = f.read_bytes(), blob(root, base_commit, rel)
        h_then = _sha(then) if then is not None else None
        if rel == r1:
            rows.append((rel, _sha(now), h_then, "R1 ITSELF -- item 1"))
            continue
        if then is None:
            verdict = "NO BLOB AT THE BASE"
        elif now == then:
            verdict = "IDENTICAL"
        elif rel == registry:
            rf = registry_faults(then.decode("utf-8"), now.decode("utf-8"), block)
            verdict = "ONLY THE A8.2 TUPLE" if not rf else "DIFFERS: " + "; ".join(rf)
        else:
            verdict = "DIFFERS"
        rows.append((rel, _sha(now), h_then, verdict))
        if verdict not in ("IDENTICAL", "ONLY THE A8.2 TUPLE"):
            faults.append(f"{rel}: {verdict}")
    return rows, faults


# ─────────────────────────────── 3. the guard, and the pre-play state ───────────────────────────────

def guard_checks(CM, pairs, block, accounted, open_gates):
    """Item 3, pure: (rows [(check, HOLDS/REFUSED/FAILS, detail)], faults). A refusal of the guard is a
    verdict; any other exception STOPS."""
    rows, faults = [], []

    def attempt(label, fn):
        try:
            v = fn()
        except CM.MatchRefused as e:
            rows.append((label, "REFUSED", str(e)))
            faults.append(f"{label}: REFUSED")
            return None
        except Exception as e:
            raise BindingStop(f"{label}: {type(e).__name__}: {e}") from None
        rows.append((label, "HOLDS", ""))
        return v

    def holds(label, ok, detail=""):
        rows.append((label, "HOLDS" if ok else "FAILS", detail))
        if not ok:
            faults.append(f"{label}: FAILS")
    binding = attempt("qualification_binding() -- A8.4 items 1-5", CM.qualification_binding)
    if binding is not None:
        rows[-1] = (rows[-1][0], "HOLDS", f"commit {binding['commit']}; corpus {binding['corpus_sha256']}; "
                    f"call list {binding['call_list_sha256']}")
    committed = attempt('_committed_plan("pilot", 0) -- committed, check_plan', lambda: CM._committed_plan("pilot", 0))
    if committed is not None:
        plan, plan_sha = committed
        rows[-1] = (rows[-1][0], "HOLDS", f"plan sha256 {plan_sha}")
        attempt("_bound_plan(plan) -- A8.4 item 6", lambda: CM._bound_plan(plan))
        got = [(p.get("pair_id"), (p.get("seeds") or {}).get("red"), (p.get("seeds") or {}).get("black"))
               for p in plan.get("pairs") or []]
        holds("the plan's pairs are the survey's pairs, ids and seeds",
              got == [(p["pair_id"], p["red"], p["black"]) for p in pairs], f"{len(got)} pairs")
    seeds = sorted(s for p in pairs for s in (p["red"], p["black"]))
    holds("the survey's pairs are exactly the block's seeds", seeds == list(range(*block)), f"block {tuple(block)}")
    holds("ACCOUNTED holds the block", tuple(block) in accounted)
    dest = CM.out_dir("pilot", 0)
    holds("the pilot destination does not exist (pre-play)", not os.path.lexists(dest), str(dest))
    holds("every gate is closed", open_gates == [], f"open: {open_gates}")
    return rows, faults


def pre_play_faults(changed, ancestor):
    """Since step 5's post-proof only the card, this script and its tests changed."""
    faults = [] if ancestor else [f"{PRE_PLAY_COMMIT} is not an ancestor of HEAD"]
    faults += [f"{p} changed since the pre-play commit" for p in changed if p not in CHANGEABLE_SINCE_PRE_PLAY]
    return faults


# ─────────────────────────────── the controls ───────────────────────────────

_C_BASE = '''"""doc"""
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


def _r1_controls():
    def r1(src):
        try:
            return not compare_r1(_C_BASE, src, named=("keep",))[2]
        except BindingStop:
            return "STOP"
    sub = _C_BASE.replace
    cases = [
        ("R1: identical", _C_BASE, True),
        ("R1: comments and blank lines only", "# c\n\n" + sub("X = 1", "X = 1  # c\n"), True),
        ("R1: a non-allowlisted function changed", sub("return 1", "return 2"), False),
        ("R1: an allowlisted function changed", sub("return 0", "return 9"), True),
        ("R1: a non-allowlisted function added", _C_BASE + "\n\ndef extra():\n    pass\n", False),
        ("R1: an allowlisted helper added", _C_BASE + "\n\ndef _read_timing():\n    pass\n", True),
        ("R1: a named function removed", sub("def keep():\n    return 1\n", ""), False),
        ("R1: a constant changed", sub("X = 1", "X = 2"), False),
        ("R1: a repository import changed", sub("helper as H", "other as H"), False),
        ("R1: a repository import added", sub("X = 1", "from . import other\nX = 1"), False),
        ("R1: a standard-library import added", sub("import os", "import os\nimport json"), True),
        ("R1: production_builder's arguments changed inside _run", sub("production_builder(a, b)",
                                                                       "production_builder(b, a)"), False),
        ("R1: a second play_game call inside _run", sub("    return play_game(g, a)",
                                                        "    play_game(g, a)\n    return play_game(g, a)"), False),
        ("R1: the play_game call removed from _run", sub("return play_game(g, a)", "return g"), False),
        ("R1: the module docstring changed", sub('"""doc"""', '"""DOC"""'), False),
        ("R1: two top-level nodes with one name STOP", _C_BASE + "\n\ndef keep():\n    return 1\n", "STOP"),
    ]
    return [(label, r1(src) == want, f"expected {want}") for label, src, want in cases]


def _tree_controls():
    """Item 2 on synthetic git repositories."""
    base = {"pkg/__init__.py": "",
            "pkg/r1.py": "from . import a\nimport json\n\n\ndef f():\n    from . import lazy\n",
            "pkg/a.py": "from .sub import b\nfrom pkg.sub import b as B2\nfrom . import reg\n",
            "pkg/sub/__init__.py": "",
            "pkg/sub/b.py": "X = 1\n",
            "pkg/lazy.py": "Y = 1\n",
            "pkg/unreached.py": "Z = 1\n",
            "pkg/reg.py": ("ACCOUNTED_SEED_INTERVALS = (\n    (100, 132),\n)\nEXPOSED_SEED_INTERVALS = ()\n"
                           "RETIRED_SEED_INTERVALS = ()\nTEST_ONLY_SEED_INTERVALS = ()\nOTHER = 1\n")}
    reg_good = base["pkg/reg.py"].replace("    (100, 132),\n", "    (100, 132),\n    # pilot\n    (200, 232),\n")
    cases = [
        ("closure: unchanged -> no difference; the unreached module is outside", {}, "SET"),
        ("closure: a module reached only by a FUNCTION's import changed", {"pkg/lazy.py": "Y = 2\n"}, False),
        ("closure: a module reached transitively changed", {"pkg/sub/b.py": "X = 2\n"}, False),
        ("closure: a subpackage __init__ changed", {"pkg/sub/__init__.py": "# x\n"}, False),
        ("closure: the package __init__ changed", {"pkg/__init__.py": "# x\n"}, False),
        ("closure: a module R1 does not reach changed", {"pkg/unreached.py": "Z = 2\n"}, True),
        ("closure: R1 reaches a module with no blob at the base",
         {"pkg/r1.py": base["pkg/r1.py"] + "from . import new\n", "pkg/new.py": "N = 1\n"}, False),
        ("registry: exactly the block appended last", {"pkg/reg.py": reg_good}, True),
        ("registry: the block appended FIRST",
         {"pkg/reg.py": base["pkg/reg.py"].replace("(\n    (100, 132),", "(\n    (200, 232),\n    (100, 132),")}, False),
        ("registry: a different tuple", {"pkg/reg.py": reg_good.replace("(200, 232)", "(200, 233)")}, False),
        ("registry: EXPOSED changed too", {"pkg/reg.py": reg_good.replace("EXPOSED_SEED_INTERVALS = ()",
                                                                          "EXPOSED_SEED_INTERVALS = ((1, 2),)")}, False),
        ("registry: another statement changed", {"pkg/reg.py": reg_good.replace("OTHER = 1", "OTHER = 2")}, False),
        ("registry: the block appended, and a comment elsewhere", {"pkg/reg.py": "# note\n" + reg_good}, True),
        ("registry: unchanged -- the block NOT appended", {"pkg/reg.py": "# note\n" + base["pkg/reg.py"]}, False),
        ("closure: importlib STOPS", {"pkg/a.py": base["pkg/a.py"] + "import importlib\n"}, "STOP"),
        ("closure: __import__() STOPS", {"pkg/sub/b.py": "X = __import__('os')\n"}, "STOP"),
        ("closure: an unresolvable relative import STOPS", {"pkg/a.py": base["pkg/a.py"] + "from . import gone\n"}, "STOP"),
        ("closure: an absolute import a sibling file could shadow STOPS", {"pkg/json.py": "J = 1\n"}, "STOP"),
    ]
    out = []
    for label, change, want in cases:
        t = pathlib.Path(tempfile.mkdtemp(prefix="r1-binding-"))
        try:
            for rel, text in base.items():
                (t / rel).parent.mkdir(parents=True, exist_ok=True)
                (t / rel).write_text(text)
            for args in (["init", "-q"], ["add", "-A"],
                         ["-c", "user.name=c", "-c", "user.email=c@c", "-c", "core.hooksPath=/dev/null",
                          "-c", "commit.gpgsign=false", "commit", "-q", "-m", "base"]):
                subprocess.run(["git", "-C", str(t), *args], capture_output=True, check=True)
            head = _git(t, "rev-parse", "HEAD").stdout.strip()
            for rel, text in change.items():
                (t / rel).parent.mkdir(parents=True, exist_ok=True)
                (t / rel).write_text(text)
            try:
                files, _ext = closure(t / "pkg/r1.py", t)
                rows, faults = compare_closure(files, t, head, (200, 232), r1="pkg/r1.py", registry="pkg/reg.py")
                got = not faults
                if want == "SET":
                    rels = sorted(r for r, *_ in rows)
                    got = "SET" if got and rels == sorted(set(base) - {"pkg/unreached.py"}) else rels
            except BindingStop:
                got = "STOP"
        finally:
            shutil.rmtree(t, ignore_errors=True)
        out.append((label, got == want, f"expected {want}"))
    return out


def _guard_controls():
    class Refused(Exception):
        pass
    pairs = [{"pair_id": f"p-{k}", "red": 300 + 2 * k, "black": 301 + 2 * k} for k in range(2)]
    plan = {"pairs": [{"pair_id": p["pair_id"], "seeds": {"red": p["red"], "black": p["black"]}} for p in pairs]}
    binding = {"commit": "c", "corpus_sha256": "x", "call_list_sha256": "y"}

    def refuse(*_a):
        raise Refused("no")

    def fake(**over):
        ns = dict(MatchRefused=Refused, qualification_binding=lambda: binding,
                  _committed_plan=lambda r, s: (plan, "sha"), _bound_plan=lambda p: binding,
                  out_dir=lambda r, s: pathlib.Path("/nonexistent/r1-binding-control"))
        ns.update(over)
        return types.SimpleNamespace(**ns)

    def g(CM, prs=pairs, block=(300, 304), acc=((300, 304),), gates=()):
        return not guard_checks(CM, prs, block, acc, list(gates))[1]
    other = [dict(p) for p in pairs]
    other[1]["black"] = 399
    return [
        ("guard: every check holds", g(fake()) is True, ""),
        ("guard: qualification_binding() refuses", g(fake(qualification_binding=refuse)) is False, ""),
        ("guard: _bound_plan() refuses", g(fake(_bound_plan=refuse)) is False, ""),
        ("guard: _committed_plan() refuses", g(fake(_committed_plan=refuse)) is False, ""),
        ("guard: a plan seed that is not the survey's", g(fake(), prs=other) is False, ""),
        ("guard: ACCOUNTED lacks the block", g(fake(), acc=((300, 302),)) is False, ""),
        ("guard: the pilot destination exists", g(fake(out_dir=lambda r, s: pathlib.Path("/"))) is False, ""),
        ("guard: an open gate", g(fake(), gates=[("m", "G")]) is False, ""),
        ("pre-play: only the card, this script and its tests changed", pre_play_faults(list(CHANGEABLE_SINCE_PRE_PLAY), True) == [], ""),
        ("pre-play: any other path changed", pre_play_faults([CARD, REGISTRY], True) != [], ""),
        ("pre-play: the pre-play commit is not an ancestor", pre_play_faults([], False) != [], ""),
        ("registry: byte-identical to the base (the block missing) is still a fault",
         registry_extra_faults([(REGISTRY, "a", "a", "IDENTICAL")], ["x"]) == ["registry: x"], ""),
        ("registry: outside the closure, still checked", registry_extra_faults([], ["x"]) == ["registry: x"], ""),
        ("registry: a closure DIFFERS row is not counted twice",
         registry_extra_faults([(REGISTRY, "a", "b", "DIFFERS: x")], ["x"]) == [], ""),
    ]


def _controls():
    return _r1_controls() + _tree_controls() + _guard_controls()


# ─────────────────────────────── the run ───────────────────────────────

def _load(rel, name, want):
    p = ROOT / rel
    if not p.is_file() or _sha(p.read_bytes()) != want:
        raise BindingRefused(f"{rel} is not the pinned file")
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _preconditions():
    """Refuses unless the record is absent, the tree clean, the commits present and every pin holds."""
    if os.path.lexists(ROOT / OUT):
        raise BindingRefused(f"{OUT} exists: the binding record runs once (create-only)")
    if _git(ROOT, "status", "--porcelain").stdout.strip():
        raise BindingRefused("the working tree is not clean: the record binds the COMMITTED state")
    for c in (BASE_COMMIT, PRE_PLAY_COMMIT):
        if _git(ROOT, "cat-file", "-e", f"{c}^{{commit}}").returncode != 0:
            raise BindingRefused(f"commit {c} is not in this repository")
    base_r1 = blob(ROOT, BASE_COMMIT, R1)
    if base_r1 is None or _sha(base_r1) != R1_BASE_SHA256:
        raise BindingRefused(f"R1 at {BASE_COMMIT} is not the qualification's pinned R1")
    P = _load(PROOF_SCRIPT, "_tactical_seed_proof", PROOF_SCRIPT_SHA256)
    S = P.load_survey()
    try:
        record = P.read_record((ROOT / P.SURVEY_RECORD).read_text(encoding="utf-8"), S)
    except P.ProofRefused as e:
        raise BindingRefused(f"the survey record: {e}") from None
    changed = [p for p in _git(ROOT, "diff", "--name-only", PRE_PLAY_COMMIT, "HEAD").stdout.splitlines() if p]
    return {"head": _git(ROOT, "rev-parse", "HEAD").stdout.strip(), "base_r1": base_r1.decode("utf-8"),
            "cur_r1": (ROOT / R1).read_text(encoding="utf-8"), "block": record["block"],
            "pairs": record["pairs"], "changed": changed,
            "ancestor": _git(ROOT, "merge-base", "--is-ancestor", PRE_PLAY_COMMIT, "HEAD").returncode == 0}


def _runtime():
    """R1 and the gate inventory, imported from the committed tree (item 3 only)."""
    sys.path.insert(0, str(ROOT))
    try:
        from scripts.GPU.alphazero import candidate_match_runner as CM
        from scripts.GPU.alphazero import gate_inventory as G
    except Exception as e:
        raise BindingStop(f"R1 or the gate inventory does not import: {type(e).__name__}: {e}") from None
    return CM, G


def _checks(ctx):
    """The three items and the pre-play state; returns every fault. BindingStop propagates."""
    faults = []
    say(f"\nPRE-PLAY STATE -- changed since {PRE_PLAY_COMMIT}:")
    for p in ctx["changed"]:
        say(f"    {p}{'' if p in CHANGEABLE_SINCE_PRE_PLAY else '   <- NOT PERMITTED'}")
    pf = pre_play_faults(ctx["changed"], ctx["ancestor"])
    say(f"  {'HOLDS' if not pf else 'FAILS -- ' + '; '.join(pf)}")
    faults += pf

    say("\n1. R1 PER TOP-LEVEL NODE (ast.dump, attributes excluded) -- base blob at "
        f"{BASE_COMMIT}, sha256 {R1_BASE_SHA256}")
    say(f"   R1 now: sha256 {_sha(ctx['cur_r1'].encode('utf-8'))}")
    rows, call_rows, f1 = compare_r1(ctx["base_r1"], ctx["cur_r1"], ALLOWLIST, NAMED_IDENTICAL, CALLS_IN_RUN)
    for k, state, why in rows:
        tag = "" if state == "IDENTICAL" else (f"   allowlisted: {why}" if why else "   <- OUTSIDE THE ALLOWLIST")
        say(f"    {state:9s} {k}{tag}")
    say(f"   {sum(r[1] == 'IDENTICAL' for r in rows)} identical, {sum(r[1] != 'IDENTICAL' for r in rows)} differing "
        f"(of those, {sum(r[1] != 'IDENTICAL' and r[2] is None for r in rows)} outside the allowlist); "
        f"named by A8.6 as identical: {len(NAMED_IDENTICAL)}")
    for name, a, b, same in call_rows:
        say(f"   inside _run, {name}(...) call nodes: {a} at the base, {b} now -- {'IDENTICAL' if same else 'NOT IDENTICAL'}")
    say(f"  {'HOLDS' if not f1 else 'FAILS -- ' + '; '.join(f1)}")
    faults += f1

    say(f"\n2. THE IMPORT CLOSURE OF R1 (every import anywhere in a module, transitively; package __init__ "
        f"files included) against {BASE_COMMIT}")
    files, ext = closure(ROOT / R1, ROOT)
    rows2, f2 = compare_closure(files, ROOT, BASE_COMMIT, ctx["block"], r1=R1, registry=REGISTRY)
    for rel, now, then, verdict in rows2:
        say(f"    {verdict:20s} {rel}\n        now {now}  base {then or 'NONE'}")
    reg_now = (ROOT / REGISTRY).read_text(encoding="utf-8")
    reg_base = blob(ROOT, BASE_COMMIT, REGISTRY)
    rf = ["the registry has no blob at the base"] if reg_base is None else \
        registry_faults(reg_base.decode("utf-8"), reg_now, ctx["block"])
    say(f"   {len(rows2)} repository files reached (R1 included); external top-level names: {', '.join(sorted(ext))}")
    say(f"   the registry ({REGISTRY}; in the closure: {any(r[0] == REGISTRY for r in rows2)}): "
        f"{'ONLY THE A8.2 TUPLE ' + str(tuple(ctx['block'])) if not rf else 'FAILS -- ' + '; '.join(rf)}")
    f2 += registry_extra_faults(rows2, rf, registry=REGISTRY)
    say(f"  {'HOLDS' if not f2 else 'FAILS -- ' + '; '.join(f2)}")
    faults += f2

    say("\n3. THE GUARD (A8.4) ON THE COMMITTED TREE, pure")
    CM, G = _runtime()
    gates = G.discover_gates()
    rows3, f3 = guard_checks(CM, ctx["pairs"], ctx["block"], accounted_of(reg_now), G.open_gates())
    for label, state, detail in rows3:
        say(f"    {state:8s} {label}{'  [' + detail + ']' if detail else ''}")
    say(f"   gates: {len(gates)} discovered, CANDIDATE_MATCH_AUTHORIZED = {CM.AUTH.CANDIDATE_MATCH_AUTHORIZED}")
    say(f"  {'HOLDS' if not f3 else 'FAILS -- ' + '; '.join(f3)}")
    return faults + f3


def verdict(faults, passed, total):
    if total == 0 or passed != total:
        return False, f"NOT BOUND -- {total - passed} control(s) failed"
    if faults:
        return False, f"NOT BOUND -- {len(faults)} difference(s); the pilot is not authorized"
    return True, "BOUND -- every node outside the allowlist, every closure file and the guard as A8.6 requires"


def _write(out):
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)       # create-only
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write("\n".join(_LINES) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def main(argv):
    if len(argv) > 1:
        raise BindingRefused("this script takes no arguments")
    ctx = _preconditions()
    out = ROOT / OUT
    say("TACTICAL CANDIDATE A8.8 STEP 6 -- THE R1 BINDING RECORD (card §18 A8.6; reads code only)")
    say("=" * 96)
    say(f"HEAD {ctx['head']}; working tree clean")
    say(f"script {SCRIPT}; card {CARD} (allowlist clarified at e8c80602)")
    say(f"the block, read from the pinned survey record via the pinned proof script: {tuple(ctx['block'])}, "
        f"{len(ctx['pairs'])} pairs")
    try:
        faults = _checks(ctx)
        results = _controls()
    except Exception as e:                       # BindingStop, or anything unforeseen: recorded, no verdict
        say(f"\nBINDING STOPPED: {type(e).__name__}: {e} -- NO VERDICT")
        _write(out)
        return 2
    say("\nCONTROLS (synthetic)")
    say("-" * 96)
    passed = 0
    for label, ok, detail in results:
        passed += ok
        say(f"  {'PASS' if ok else 'FAIL':6s} {label}{'  [' + detail + ']' if detail and not ok else ''}")
    say(f"\ncontrols passed: {passed}/{len(results)}")
    ok, text = verdict(faults, passed, len(results))
    say(f"BINDING VERDICT: {text}")
    _write(out)
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except BindingRefused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        sys.exit(4)
