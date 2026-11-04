"""H-TB teacher setup, part 1: re-acquire twixtbot-ui at the pinned revision and verify it (2026-11-04).

Approved by the user's decision of 2026-11-04 (isolated dependency setup only). Standard library only. It never
imports anything from the checkout, never initializes a model and never reads the game corpus.

  * refuses unless the checkout destination and both outputs are absent (fresh, create-only);
  * clones, then detaches at the pinned commit;
  * verifies HEAD, a clean status, and EVERY tracked file's content (git hash-object == the HEAD tree's blob id);
  * verifies the three model pins and records the sha256 of every model file;
  * retains the license/notice files (copied create-only next to this script) and checks LICENSE against the
    sha256 recorded by the October setup (evidence/external-engine-learning-setup/02_setup_check.json).

    python3 -I 01_checkout.py <destination>      # writes 02_checkout.json and 03_*.txt beside this script
"""
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys

REPO = "https://github.com/stevens68/twixtbot-ui"
PIN = "83749f230a0bae1766b46a05bfde0ed87f0a9a0a"
MODEL_PINS = {
    "model/pb/variables/variables.data-00000-of-00001": "1958f8476e9d56cbb87fa570db88f9cc9d389b30e20571fd66e8e779a4cefbab",
    "model/pb/saved_model.pb": "e0c2b882bc97c4661ac92af47ae5ef78443e57b42d0cb7c9c3a8cb7e8d01993b",
    "model/pb/variables/variables.index": "1d4073c4da30e515a984faef2c3cc9c8b8ba402353a01933c8bb80e1f279f8cd",
}
OCTOBER_LICENSE_SHA256 = "cef0d970b9a3dcdc2d3b7bc2ecee97a67d57cc1d1287d1ea5e13fe86b6dbd36b"
HERE = pathlib.Path(__file__).resolve().parent


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(dest, *args):
    return subprocess.run(["git", "-C", str(dest), *args], capture_output=True, text=True, check=True).stdout


def main(dest):
    dest = pathlib.Path(dest).expanduser().resolve()
    out_json = HERE / "02_checkout.json"
    for p in (dest, out_json):
        if os.path.lexists(p):
            raise SystemExit(f"REFUSED: {p} already exists")
    subprocess.run(["git", "clone", "--quiet", "--no-checkout", REPO, str(dest)], check=True)
    git(dest, "checkout", "--quiet", "--detach", PIN)
    head = git(dest, "rev-parse", "HEAD").strip()
    status = git(dest, "status", "--porcelain").strip()
    if head != PIN or status:
        raise SystemExit(f"FAILED: HEAD {head}, status {status!r}")

    tree = {}
    for line in git(dest, "ls-tree", "-r", "HEAD").splitlines():
        meta, path = line.split("\t", 1)
        mode, kind, blob = meta.split()
        if kind == "blob":
            tree[path] = (mode, blob)
    paths = sorted(tree)
    actual = git(dest, "hash-object", "--no-filters", "--", *paths).split()
    mismatched = [p for p, a in zip(paths, actual) if tree[p][1] != a]
    if mismatched:
        raise SystemExit(f"FAILED: content differs from the pinned tree: {mismatched[:5]}")
    manifest = {p: {"blob": tree[p][1], "sha256": sha256(dest / p)} for p in paths}

    model_bad = {rel: manifest[rel]["sha256"] for rel, want in MODEL_PINS.items() if manifest[rel]["sha256"] != want}
    if model_bad:
        raise SystemExit(f"FAILED: model pins differ: {model_bad}")

    notices = [p for p in paths if pathlib.PurePosixPath(p).name.upper().startswith(("LICENSE", "NOTICE", "COPYING"))]
    if not notices:
        raise SystemExit("FAILED: the checkout carries no license or notice file")
    kept = []
    for i, rel in enumerate(notices):
        target = HERE / f"03_notice_{i}_{rel.replace('/', '__')}.txt"
        with open(dest / rel, "rb") as src, open(target, "xb") as dst:
            shutil.copyfileobj(src, dst)
        kept.append({"path": rel, "sha256": manifest[rel]["sha256"], "retained_as": target.name})

    record = {
        "kind": "htb-teacher-setup-checkout/1",
        "repo": REPO, "commit": head, "destination": str(dest),
        "tracked_files": len(paths), "content_verified": True,
        "model_pins_verified": sorted(MODEL_PINS),
        "model_files": {p: manifest[p] for p in paths if p.startswith("model/")},
        "notices": kept,
        "license_matches_october_setup": manifest.get("LICENSE", {}).get("sha256") == OCTOBER_LICENSE_SHA256,
        "manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        "manifest": manifest,
        "git": subprocess.run(["git", "--version"], capture_output=True, text=True).stdout.strip(),
    }
    with open(out_json, "x") as fh:
        json.dump(record, fh, indent=1, sort_keys=True)
    print(json.dumps({k: v for k, v in record.items() if k != "manifest"}, indent=1, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1])
