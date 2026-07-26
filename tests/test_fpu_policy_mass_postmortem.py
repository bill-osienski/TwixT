"""Pure contracts for the read-only policy-mass control-flip postmortem."""

from scripts.GPU.alphazero import postmortem_fpu_policy_mass_controls as pm


def _feature(*, move, prior, rank, q, value, top_share, eff, replies,
             collapsed=False, lock_in=False, mass=0.7, stabilization=42):
    return {
        "selected_move_id": move,
        "selected_move": pm.move_label(move),
        "selected_move_prior": prior,
        "selected_move_prior_rank": rank,
        "q_parent_final": q,
        "root_value_stm": value,
        "top_share": top_share,
        "effective_children": eff,
        "replies": replies,
        "collapsed": collapsed,
        "lock_in": lock_in,
        "explored_mass": mass,
        "explored_mass_at_stabilization": mass - 0.1,
        "stabilization_sim": stabilization,
    }


def test_paired_control_row_carries_required_pairs_and_lower_prior_flip():
    manifest = {
        "canonical_position_sha1": "a" * 40,
        "game_idx": "17",
        "position_ply": "88",
        "ply_bucket": "late",
        "branching_band": "b400_plus",
        "side": "red",
        "role": "control",
    }
    off = _feature(move=25, prior=0.20, rank=2, q=0.1, value=0.1,
                   top_share=0.4, eff=12.0, replies=21)
    r0 = _feature(move=26, prior=0.05, rank=8, q=-0.2, value=-0.2,
                  top_share=0.8, eff=4.0, replies=3, collapsed=True,
                  lock_in=True)

    row = pm.paired_control_row(manifest, off, r0)

    assert row["control_flip_to_lower_prior"] is True
    assert row["absolute_off_selected_move"] == "(1,1)"
    assert row["r0_selected_move"] == "(1,2)"
    assert row["root_value_delta_r0_minus_absolute_off"] == -0.3
    assert row["top_share_delta_r0_minus_absolute_off"] == 0.4
    assert row["effective_children_delta_r0_minus_absolute_off"] == -8.0
    assert row["reply_count_delta_r0_minus_absolute_off"] == -18
    assert row["collapse_transition"] == "uncollapsed_to_collapsed"
    assert row["lock_in_transition"] == "unlocked_to_locked"


def test_summary_separates_flipped_and_nonflipped_and_is_canonical():
    base = {
        "canonical_position_sha1": "x" * 40, "game_idx": "1",
        "position_ply": "31", "phase": "early_mid", "ply_bucket": "early_mid",
        "branching_band": "b200_299", "side": "black",
        "absolute_off_q_parent_final": 0.2,
        "r0_q_parent_final": 0.1,
        "root_value_delta_r0_minus_absolute_off": -0.1,
        "top_share_delta_r0_minus_absolute_off": 0.2,
        "effective_children_delta_r0_minus_absolute_off": -3.0,
        "reply_count_delta_r0_minus_absolute_off": -4,
        "absolute_off_selected_move_prior_rank": 1,
        "r0_selected_move_prior_rank": 5,
    }
    flipped = {**base, "control_flip_to_lower_prior": True}
    retained = {**base, "canonical_position_sha1": "y" * 40,
                "control_flip_to_lower_prior": False}

    summary = pm.summarize_controls([flipped, retained])

    assert summary["n_controls"] == 2
    assert summary["n_lower_prior_flips"] == 1
    assert summary["flip_rate"] == 0.5
    assert summary["by_phase"]["early_mid"] == {"flipped": 1, "total": 2, "rate": 0.5}
    assert summary["flipped_vs_nonflipped"]["flipped"]["n"] == 1
    assert pm.canonical_json(summary) == pm.canonical_json(summary)
