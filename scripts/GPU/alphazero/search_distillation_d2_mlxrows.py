"""SEARCH DISTILLATION D2 -- SYNTHETIC ROW-COUNT PROBE: the gated entry for the standalone
`mlx_matmul_rows_probe.py`. BUILT; run only through its wrapper.

Card: `docs/superpowers/2026-10-23-search-distillation-d2-synthetic-rows-probe-card.md`, pinned below by sha256, as is
the generator. Nothing runs on import. `run` refuses unless
`search_distillation_d2_mlxrows_authorization.SEARCH_DISTILLATION_D2_MLXROWS_AUTHORIZED` is open (read at call time),
the card and the generator are the pinned bytes, and neither output exists. It then hands the generator its two
create-only directories and returns its exit code. The generator itself stays standalone (standard library, NumPy,
MLX), so it can be carried out of this repository as a check for a repair.

    python -m scripts.GPU.alphazero.search_distillation_d2_mlxrows run
"""
from __future__ import annotations

import datetime
import hashlib
import os
import pathlib
import sys
from typing import Sequence

from . import mlx_matmul_rows_probe as P
from . import search_distillation_d2_mlxrows_authorization as GATE

ROOT = pathlib.Path(__file__).resolve().parents[3]
CARD = "docs/superpowers/2026-10-23-search-distillation-d2-synthetic-rows-probe-card.md"
CARD_SHA256 = "6177c3de27f305ce78d10d7de775ba92d2784370e8d5c6fda0fde1df62ed30cc"
GENERATOR = "scripts/GPU/alphazero/mlx_matmul_rows_probe.py"
GENERATOR_SHA256 = "1217bc8e48589b077bbe0962564ffb5c0298326f96de6658ae39a83cf739bb57"
EVIDENCE_DIR = "docs/superpowers/evidence/search-distillation-d2-mlxrows"
OUTPUT_PARENT = pathlib.Path.home() / "twixt-cache" / "search-distillation-d2"          # outside the repository
RECORD_KIND = P.RECORD_KIND
EXIT_COMPLETED, EXIT_USAGE, EXIT_VOID, EXIT_REFUSED = P.EXIT_COMPLETED, P.EXIT_USAGE, P.EXIT_VOID, P.EXIT_REFUSED


class Refused(Exception):
    """The run never started: gate closed, card or generator not the pinned bytes, or an output exists."""


def _sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run() -> int:
    """THE PUBLIC ENTRY. No seam: the frozen generator, the pinned card, the two create-only directories."""
    if GATE.SEARCH_DISTILLATION_D2_MLXROWS_AUTHORIZED is not True:
        raise Refused("SEARCH_DISTILLATION_D2_MLXROWS_AUTHORIZED is closed: the probe is not authorized")
    if _sha(ROOT / CARD) != CARD_SHA256:
        raise Refused(f"{CARD} is not the pinned card")
    if _sha(pathlib.Path(P.__file__)) != GENERATOR_SHA256 or pathlib.Path(P.__file__).resolve() != ROOT / GENERATOR:
        raise Refused(f"{GENERATOR} is not the pinned generator")
    out = ROOT / EVIDENCE_DIR
    if os.path.lexists(out):
        raise Refused(f"{out} exists: the run is one-shot")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    example = OUTPUT_PARENT / f"mlxrows-{stamp}"
    if not OUTPUT_PARENT.is_dir() or os.path.lexists(example):
        raise Refused(f"{OUTPUT_PARENT} is not a directory, or {example} exists")
    return P.main(["--out", str(out), "--example-dir", str(example)])


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv != ["run"]:
        print("usage: python -m scripts.GPU.alphazero.search_distillation_d2_mlxrows run", file=sys.stderr)
        return EXIT_USAGE
    try:
        return run()
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
