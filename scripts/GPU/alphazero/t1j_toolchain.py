"""Where the T1j toolchain lives, and proof it is the right one.

NO EXECUTION. This module starts no JVM, loads no model, draws no seed and
queries nothing. It resolves a filesystem root and hashes files.

WHY IT EXISTS. The JDK and the T1j jar were previously pinned at a path inside a
session scratchpad under /private/tmp. That directory was cleaned, the artifacts
went with it, and 41 qualification tests turned red at once -- correctly, because
they fail closed on a missing pinned component, but for a reason that had nothing
to do with the code under test.

THE RULES, in order of importance:

  * THE ARTIFACTS ARE NOT COMMITTED. Only `t1j_toolchain_lock.json` is: source
    URLs, expected hashes, version and relative layout.
  * The local root comes from an EXPLICIT setting -- an argument, else the
    environment variable, else the recorded default -- and the resolver always
    reports WHICH. A default that announces itself is not a silent fallback.
  * A root under /tmp or /private/tmp is REFUSED outright, whatever supplied it.
    That is the failure this module exists to prevent, so it is not overridable.
  * Nothing is returned unverified. `verified_paths` hashes the jar and every
    pinned JDK component before handing back a single path.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Optional, Tuple

#: The committed identity record. The artifacts it describes are NOT in the repo.
LOCK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "t1j_toolchain_lock.json")

#: The setting. An explicit argument beats it; it beats the recorded default.
ENV_VAR = "TWIXT_T1J_TOOLCHAIN_ROOT"

#: The durable location. Chosen deliberately outside the repository (the JDK is
#: ~186 MB) and outside any temp directory.
DEFAULT_ROOT = os.path.expanduser(
    "~/Library/Application Support/TwixT_Game/toolchains/t1j-e1")

#: Prefixes that may never host the toolchain, from any source.
EPHEMERAL_PREFIXES = ("/tmp", "/private/tmp", "/var/tmp", "/private/var/tmp")


class ToolchainError(Exception):
    """The toolchain is absent, misplaced, or not the pinned one."""


def load_lock(path: str = LOCK_PATH) -> Dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except OSError as e:
        raise ToolchainError(f"cannot read the toolchain lock: {e}") from None


def _reject_ephemeral(root: str, source: str) -> None:
    real = os.path.realpath(root)
    for bad in EPHEMERAL_PREFIXES:
        if real == bad or real.startswith(bad + os.sep):
            raise ToolchainError(
                f"toolchain root {root!r} (from {source}) resolves under {bad!r}, an "
                f"ephemeral directory. That is exactly how the toolchain was lost "
                f"once; it is refused rather than warned about.")


def resolve_root(root: Optional[str] = None) -> Tuple[str, str]:
    """Return (root, source). Explicit > environment > recorded default.

    The source is returned, not logged and discarded, so a caller can never be
    unsure which location it got. Refuses an ephemeral or absent root.
    """
    if root is not None:
        chosen, source = root, "explicit"
    elif os.environ.get(ENV_VAR):
        chosen, source = os.environ[ENV_VAR], "environment"
    else:
        chosen, source = DEFAULT_ROOT, "recorded-default"
    _reject_ephemeral(chosen, source)
    if not os.path.isdir(chosen):
        raise ToolchainError(
            f"toolchain root {chosen!r} (from {source}) does not exist. Acquire the "
            f"artifacts recorded in {os.path.basename(LOCK_PATH)}; nothing is "
            f"substituted for a missing root.")
    return chosen, source


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check(path: str, want: str, what: str) -> None:
    if not os.path.isfile(path):
        raise ToolchainError(f"{what} missing at {path}")
    got = _sha256(path)
    if got != want:
        raise ToolchainError(f"{what} sha256 {got} != pinned {want}")


def verified_paths(root: Optional[str] = None,
                   lock: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Hash-verify the toolchain, THEN return its paths. Never the other way."""
    lock = load_lock() if lock is None else lock
    chosen, source = resolve_root(root)
    jar = os.path.join(chosen, lock["layout"]["jar"])
    jdk_home = os.path.join(chosen, lock["layout"]["jdk_home"])

    want_bytes = lock["jar"].get("bytes")
    if want_bytes is not None:
        if not os.path.isfile(jar):
            raise ToolchainError(f"t1j.jar missing at {jar}")
        size = os.path.getsize(jar)
        if size != want_bytes:
            raise ToolchainError(f"t1j.jar is {size} bytes, expected {want_bytes}")
    _check(jar, lock["jar"]["sha256"], "t1j.jar")
    verified = 1
    for rel, want in sorted(lock["jdk"]["components"].items()):
        _check(os.path.join(jdk_home, rel), want, f"pinned JDK component {rel}")
        verified += 1
    return {"root": chosen, "source": source, "jar": jar, "jdk_home": jdk_home,
            "verified": verified}
