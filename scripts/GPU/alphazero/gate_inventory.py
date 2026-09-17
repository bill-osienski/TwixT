"""THE AUTHORIZATION-GATE INVENTORY, DERIVED FROM SOURCE.

🔴 WHY THIS MODULE EXISTS. Three pre-run checkers each kept their OWN hand-typed
list of gates. They disagreed: H2's said seven, the pilot's said eight, the
study's said ten. Reports quoted whichever one was nearest and said "all EIGHT
gates are False" for weeks. Ten was right.

The number was never the defect. **A hand-kept list cannot see a gate nobody
remembered to add to it**, which is the one case a gate count exists to catch --
and it is the programme's recurring shape: a check that exists but does not bind.
Two gates were missed for the most ordinary reason, that their qualifications had
already run; a RETIRED gate is still a closed authorization constant in the
source and still belongs in the count.

So the inventory is ENUMERATED FROM THE SOURCE by AST and every checker imports
it. `tests/test_gate_inventory.py` pins the expected set, so adding or removing a
gate is a deliberate edit and never a silent drift.

Nothing here imports the gated modules. Reading a gate must never be able to run
one, so discovery parses text and never executes it.
"""
from __future__ import annotations

import ast
import pathlib
from typing import Dict, List, Tuple

#: the package whose module-level constants are gates
_PKG = pathlib.Path(__file__).resolve().parent
_ROOT = _PKG.parents[2]


def discover_gates(root: pathlib.Path | None = None) -> Dict[Tuple[str, str], bool]:
    """Map ``(module_name, constant_name) -> value`` for every gate in the source.

    A gate is a **module-level** assignment of a **bool** to a name containing
    ``AUTHORIZED``.

    * *module-level*, because a local named ``..._AUTHORIZED`` inside a function
      is not a gate and counting one would make the total meaningless;
    * *bool*, because ``EXIT_UNAUTHORIZED = 5`` is an exit code. Fourteen of
      those exist and none is an authorization.

    🔑 PARSED, NEVER IMPORTED. `ast.parse` does not execute the module, so taking
    an inventory of the gates cannot trip one.
    """
    base = pathlib.Path(root) if root is not None else _ROOT / "scripts"
    out: Dict[Tuple[str, str], bool] = {}
    for path in sorted(base.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if not (isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, bool)):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and "AUTHORIZED" in target.id:
                    out[(path.stem, target.id)] = node.value.value
    return out


def gates() -> Tuple[Tuple[str, str], ...]:
    """The ``(module, constant)`` pairs, sorted, for a checker to iterate."""
    return tuple(sorted(discover_gates()))


def gate_count() -> int:
    """How many gates the source actually contains. **Never write this number.**"""
    return len(discover_gates())


def open_gates() -> List[Tuple[str, str]]:
    """Every gate whose source value is not ``False``. Empty is the shut state."""
    return sorted(k for k, v in discover_gates().items() if v is not False)
