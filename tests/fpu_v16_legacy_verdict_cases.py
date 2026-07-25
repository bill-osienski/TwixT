"""Frozen `dev_safety_verdict` input fixtures, captured BEFORE the v17 keyword
parameterization (`stratum_gate`, `lockin_margin`).

`tests/golden/fpu_v16_dev_safety_verdicts.json` holds the outputs these produced
on the pre-change code. Replaying these inputs through the parameterized
function must reproduce that file exactly -- reasons, rejected flag, and every
metric -- which is what proves the v17 change is backward-compatible rather than
merely "the tests still pass".

The spread deliberately covers each gate and both sides of the boundaries that
have one, plus the two pre-existing kwargs (`include_stratum_census`,
`stratum_key="ply_bucket"`).
"""


def target(band, ply_bucket, *, new_collapse=False, lock_in=False,
           mover_delta=0.0, eff=0.0, tsi=0.0):
    return {"role": "target", "band": band, "ply_bucket": ply_bucket,
            "new_collapse": new_collapse, "lock_in": lock_in,
            "mover_delta": mover_delta, "eff_children_reduction": eff,
            "top_share_inc": tsi}


def control(*, flip=False, mover_delta=0.0):
    return {"role": "control", "mover_delta": mover_delta,
            "control_flip_to_lower_prior": flip}


CASES = {
    "clean": [target("b200_299", "late") for _ in range(25)] + [control()] * 10,
    "target_collapse_trips": (
        [target("b200_299", "late", new_collapse=True) for _ in range(2)]
        + [target("b200_299", "late") for _ in range(23)]),
    "band_subgate_trips": (
        [target("b200_299", "late", new_collapse=True) for _ in range(3)]
        + [target("b200_299", "late") for _ in range(17)]
        + [target("b300_399", "midgame") for _ in range(40)]),
    "lockin_trips": [target("b200_299", "late", lock_in=(i < 4))
                     for i in range(25)],
    "lockin_at_margin": [target("b200_299", "late", lock_in=(i < 2))
                         for i in range(25)],
    "p95_mover_trips": [target("b200_299", "late", mover_delta=0.9 if i < 3 else 0.0)
                        for i in range(25)],
    "compound_trips": [target("b200_299", "late", eff=0.6, tsi=0.2)
                       for _ in range(25)],
    "compound_one_side_only": [target("b200_299", "late", eff=0.6, tsi=0.1)
                               for _ in range(25)],
    "control_flip_trips": [control(flip=(i < 3)) for i in range(10)],
    "control_p95_trips": [control(mover_delta=0.9 if i < 2 else 0.0)
                          for i in range(10)],
    "mixed": ([target("b200_299", "late", new_collapse=(i == 0), lock_in=(i < 3),
                      mover_delta=0.1 * i, eff=0.3, tsi=0.05) for i in range(22)]
              + [control(flip=(i == 0), mover_delta=0.05 * i) for i in range(12)]),
}
