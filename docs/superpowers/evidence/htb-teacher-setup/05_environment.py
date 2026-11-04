"""H-TB teacher setup, part 2: an isolated CPU teacher environment (2026-11-04).

Approved by the user (2026-11-04): Python 3.13.15 from astral-sh/python-build-standalone release 20260929,
`cpython-3.13.15+20260929-aarch64-apple-darwin-install_only.tar.gz`, sha256 003d459a…, extracted into the fresh
~/twixt-cache/htb-teacher/python; a separate venv beside it; packages = EXACTLY the October baseline freeze
(evidence/external-engine-learning-setup/03_venv_freeze.txt, TensorFlow 2.21.0). Global PATH, Homebrew, the pip
cache and the project's .venv are untouched; pip runs with no config file, no cache and no user site.

Never imports TensorFlow (or any installed package) and never initializes a model or reads the corpus: versions and
licenses are read through importlib.metadata in the new interpreter.

    <any python3> -I 05_environment.py           # writes 06_requirements.txt and 07_environment.json beside this file
"""
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tarfile
import urllib.request

BASE = pathlib.Path.home() / "twixt-cache" / "htb-teacher"
PY_DIR, VENV, DL, WHEELS = BASE / "python", BASE / "teacher-venv", BASE / "downloads", BASE / "wheelhouse"
ARCHIVE = "cpython-3.13.15+20260929-aarch64-apple-darwin-install_only.tar.gz"
ARCHIVE_URL = f"https://github.com/astral-sh/python-build-standalone/releases/download/20260929/{ARCHIVE.replace('+', '%2B')}"
ARCHIVE_SHA256 = "003d459a75ff6949a6590812b1e02a65e849b5b2a9f47c421010138ec643a11c"
WANT_PYTHON = "3.13.15"
HERE = pathlib.Path(__file__).resolve().parent
BASELINE = HERE.parent / "external-engine-learning-setup" / "03_venv_freeze.txt"
ENV = {"PATH": "/usr/bin:/bin", "HOME": str(pathlib.Path.home()), "PIP_CONFIG_FILE": os.devnull,
       "PIP_NO_CACHE_DIR": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1", "PYTHONNOUSERSITE": "1"}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(*args, **kw):
    return subprocess.run([str(a) for a in args], env=ENV, capture_output=True, text=True, check=True, **kw).stdout


def main():
    outs = (HERE / "06_requirements.txt", HERE / "07_environment.json")
    for p in (PY_DIR, VENV, DL, WHEELS, *outs):
        if os.path.lexists(p):
            raise SystemExit(f"REFUSED: {p} already exists")
    rec = {"kind": "htb-teacher-setup-environment/1"}

    # 1. the interpreter archive, verified before extraction
    DL.mkdir(parents=True)
    arch = DL / ARCHIVE
    with urllib.request.urlopen(ARCHIVE_URL, timeout=120) as r, open(arch, "xb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    got = sha256(arch)
    if got != ARCHIVE_SHA256:
        raise SystemExit(f"FAILED: archive sha256 {got} != published {ARCHIVE_SHA256}")
    with tarfile.open(arch) as t:
        names = t.getnames()
        if not all(n == "python" or n.startswith("python/") for n in names):
            raise SystemExit("FAILED: the archive has members outside python/")
        t.extractall(BASE, filter="data")
    py = PY_DIR / "bin" / "python3.13"
    version = run(py, "-I", "-c", "import sys; print(sys.version.split()[0])").strip()
    if version != WANT_PYTHON:
        raise SystemExit(f"FAILED: interpreter is {version}, not {WANT_PYTHON}")
    rec["interpreter"] = {
        "distributor": "astral-sh/python-build-standalone", "release": "20260929", "archive": ARCHIVE,
        "url": ARCHIVE_URL, "sha256": got, "published_sha256_matches": True, "members": len(names),
        "path": str(py), "version": version,
        "sys_version": run(py, "-I", "-c", "import sys; print(sys.version)").strip(),
        "platform": run(py, "-I", "-c", "import platform; print(platform.platform())").strip(),
        "note": "Same version as the October 3.13.15; a different distributor. Equal versions do not establish an "
                "equivalent interpreter (the October build's source was never recorded).",
    }

    # 2. the separate venv
    run(py, "-I", "-m", "venv", VENV)
    vpy = VENV / "bin" / "python"
    rec["venv"] = {"path": str(VENV), "python": run(vpy, "-I", "-c", "import sys; print(sys.version.split()[0])").strip(),
                   "pyvenv_cfg": (VENV / "pyvenv.cfg").read_text(),
                   "pip": run(vpy, "-I", "-m", "pip", "--version").strip()}

    # 3. exactly the baseline's packages: wheels only, no dependency resolution, digests checked against PyPI
    lines = [l.strip() for l in BASELINE.read_text().splitlines() if "==" in l]
    req = HERE / "06_requirements.txt"
    with open(req, "x") as f:
        f.write("\n".join(lines) + "\n")
    WHEELS.mkdir()
    run(vpy, "-I", "-m", "pip", "download", "--only-binary=:all:", "--no-deps", "-r", req, "-d", WHEELS)
    wheels = sorted(WHEELS.iterdir())
    published, bad = {}, []
    for name, ver in (l.split("==") for l in lines):
        with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{ver}/json", timeout=60) as r:
            for u in json.load(r)["urls"]:
                published[u["filename"]] = u["digests"]["sha256"]
    rec["wheels"] = []
    for w in wheels:
        d = sha256(w)
        ok = published.get(w.name) == d
        rec["wheels"].append({"file": w.name, "sha256": d, "matches_pypi": ok})
        if not ok:
            bad.append(w.name)
    if bad or len(wheels) != len(lines):
        raise SystemExit(f"FAILED: {len(wheels)} wheels for {len(lines)} requirements; digest mismatches {bad}")
    run(vpy, "-I", "-m", "pip", "install", "--no-index", "--find-links", WHEELS, "--no-deps", "-r", req)
    rec["pip_check"] = run(vpy, "-I", "-m", "pip", "check").strip()

    # 4. versions and licenses by metadata only (nothing installed is imported)
    meta = json.loads(run(vpy, "-I", "-c", (
        "import importlib.metadata as m, json, sys\n"
        "out = {d.metadata['Name']: {'version': d.version, 'license': d.metadata.get('License-Expression')"
        " or d.metadata.get('License') or '', 'classifiers': [c for c in (d.metadata.get_all('Classifier') or [])"
        " if c.startswith('License')]} for d in m.distributions()}\n"
        "print(json.dumps({'dists': out, 'tensorflow_imported': 'tensorflow' in sys.modules}))")))
    rec["installed"] = meta["dists"]
    rec["tensorflow_imported"] = meta["tensorflow_imported"]
    want = {n.lower().replace("_", "-"): v for n, v in (l.split("==") for l in lines)}
    have = {n.lower().replace("_", "-"): d["version"] for n, d in meta["dists"].items() if n.lower() != "pip"}
    rec["matches_baseline_freeze"] = have == want
    rec["baseline_freeze"] = {"path": str(BASELINE.relative_to(HERE.parents[2])), "sha256": sha256(BASELINE),
                              "n": len(lines)}
    if have != want:
        rec["baseline_differences"] = {k: (want.get(k), have.get(k)) for k in sorted(set(want) | set(have))
                                       if want.get(k) != have.get(k)}
    rec["project_venv_untouched_check"] = "see 08_project_env_after.txt"
    with open(HERE / "07_environment.json", "x") as fh:
        json.dump(rec, fh, indent=1, sort_keys=True)
    print(json.dumps({k: v for k, v in rec.items() if k not in ("installed", "wheels")}, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
