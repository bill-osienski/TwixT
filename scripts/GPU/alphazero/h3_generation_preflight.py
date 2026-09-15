"""H3 FULL STUDY — the OPENING-GENERATION PREFLIGHT. Constructs, never invokes.

🔴 IT LIVES IN ITS OWN MODULE, AND THAT IS STRUCTURAL, NOT TIDINESS.
`test_NO_CONTROL_DELETES_AN_AUTHORIZATION_CHECK` identifies a module's production
seam as THE LATEST-DEFINED FUNCTION THAT RESOLVES THE TOOLCHAIN, and requires the
containment boundary inside it. This preflight resolves the toolchain too — that
is the point of it — so sitting in `h3_study_generator` it became "the seam" and
the real seam's boundary stopped being visible. The invariant refused, correctly.

It cannot carry the boundary itself: the boundary refuses inside a test process,
and this must be runnable from the test suite. So it moves out, leaving the
generator with exactly one function that resolves the toolchain — the gated,
bounded one.

🔑 IT IS DELIBERATELY UNGATED. It generates nothing, plays nothing and asks for
no move; gating it would only discourage running it. What it must never do is
move, and a control injects exactly that.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from . import h3_study_generator as GEN
from . import h3_study_rules as RULES


# ═══════════════════════ the PREFLIGHT: construct, never invoke ════════════
class _StubEvaluator:
    """Stands in for the checkpoint that will be LOADED, carrying the identity
    tags `build_reference_agent` checks. Retyped on purpose: reading them off the
    task would leave that check comparing a value with itself."""
    _g3_reference = "calib020_0001"
    _g3_sha1 = "209cf2d4fd24a48553d259dd71b4954867b9473e"


def preflight_movers(*, evaluator=None, classes: Optional[str] = None
                     ) -> Dict[str, Any]:
    """Build BOTH movers through the REAL production path and NEVER ask for a move.

    🔑 WHAT IS REAL HERE: `t1j_toolchain.verified_paths` (which hashes the jar and
    the JDK and refuses a mismatch, and starts no process), a real
    `e4_screen_integration.T1jRuntime`, a real `T1jAgent`, and a real
    `twixtbot_g3_reference.build_reference_agent` — the call that VOIDed H2's
    attempt 1 and which the pilot then found a second way to fail.

    🔑 WHAT IS NOT: no model is loaded (a stub evaluator carries the identity
    tags), nothing is compiled, no JVM starts, and neither agent is CALLED. The
    T1j classes directory is deliberately a path that need not exist: compiling
    is a production act and this is a preflight.

    🔴 IT DOES NOT TOUCH THE GATE, and it must not: it generates nothing, so
    gating it would only discourage running it. What it must never do is move.
    """
    from . import e4_screen_integration as INT
    from . import t1j_toolchain as TC
    import os as _os

    cfg = RULES.generation_config()                # refuses an argmax config
    paths = TC.verified_paths()
    runtime = INT.T1jRuntime(
        java=_os.path.join(paths["jdk_home"], "bin", "java"), jar=paths["jar"],
        classes=classes or (GEN.DEFAULT_OUT + ".t1j_classes"),
        ply_cap=RULES.t1j_ply_cap(), timeout_s=RULES.PER_CALL_TIMEOUT_S)
    movers = GEN.production_movers(evaluator=evaluator or _StubEvaluator(),
                               runtime=runtime, config=cfg)

    built = []
    for order in (RULES.ORDER_INCUMBENT_FIRST, RULES.ORDER_T1J_FIRST):
        inc_colour = GEN.incumbent_colour(order)
        t1j_colour = "black" if inc_colour == "red" else "red"
        seed = RULES.attempt_seed(RULES.GEN_SEED_CO_PRODUCED, 0, 0)
        ctx = movers["new_context"]()
        inc = movers["incumbent_agent"](seed=seed, colour=inc_colour)
        t1j = movers["t1j_agent"](colour=t1j_colour, ctx=ctx)
        if getattr(inc, "seed", None) != seed:
            raise GEN.H3GenerationError(
                f"the incumbent agent carries seed {getattr(inc, 'seed', None)}, "
                f"not the attempt seed {seed}")
        if t1j.colour != t1j_colour or t1j.depth != RULES.t1j_depth():
            raise GEN.H3GenerationError(
                f"the T1j agent is {t1j.colour} at depth {t1j.depth}")
        if t1j.runtime is not runtime:
            raise GEN.H3GenerationError("the T1j agent holds a different runtime")
        built.append({"order": order, "incumbent_colour": inc_colour,
                      "t1j_colour": t1j_colour, "seed": seed,
                      "incumbent_seed": getattr(inc, "seed", None),
                      "readout": getattr(getattr(inc, "readout", None), "mode",
                                         None),
                      "t1j_depth": t1j.depth, "moves_made": t1j.moves_made})
    return {"built": built, "config_pins": GEN.config_pins(cfg),
            "toolchain": GEN.toolchain_identity(paths), "runtime": runtime,
            "movers": movers}
