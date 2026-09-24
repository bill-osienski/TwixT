"""The E4 endpoint screen's reference-agent construction. NO GAMES HERE.

Constructing an agent plays nothing and consumes no seed. This module exists so
that the E4 screen, if it is ever authorized, cannot invent its own reference
path: it delegates to `twixtbot_g3_reference.build_reference_agent`, the
construction already qualified for the twixtbot G3 calibration, and refuses any
task that does not carry the fields that path requires.

WHAT THE E4 PREFLIGHT ATTEMPT 2 GOT WRONG, AND THIS FIXES
---------------------------------------------------------
Attempt 2's schedule recorded a `seed` and stopped there. Nothing bound the seed
to the two RNG streams that actually decide our moves, and the tasks were missing
two fields `build_reference_agent` demands -- `reference_sha1` and
`anchor_colour` -- so every task would have been refused at construction time.

Attempt 2 also declared ``dirichlet_eps: 0.0`` as an override. THAT IS NOT HOW
NOISE IS DISABLED ON THIS PATH. `eval_runner.cfg_from` builds its `MCTSConfig`
without passing `dirichlet_eps` at all, so the field keeps its dataclass default
of 0.25; root noise is suppressed instead at the call site, by
``search_with_root(state, add_noise=False)`` in `SeededReferenceAgent.__call__`.
The declaration was cosmetic. It is replaced here by a statement of the real
mechanism, and by a check that reads the mechanism rather than the declaration.

TWO SEED REGISTRIES, AND TWO VALIDATION QUESTIONS
-------------------------------------------------
`EXPOSED_SEED_INTERVALS` records EXPERIMENTAL EXPOSURE: seeds drawn from OUTSIDE
the permanently unschedulable test namespace, so a seed a schedule could have used
has been struck off. `RETIRED_SEED_INTERVALS` records what MAY NOT BE USED -- a
rule about the future.
They are kept apart because merging them would overstate the record: the
canonical screen drew from 24 of its 32 seeds and skipped 8 undrawn, and calling
all 32 "exposed" claims a draw that never happened. Both refuse execution.

A THIRD list, `TEST_ONLY_SEED_INTERVALS`, is neither: it is a band reserved for
the seeds tests draw from, ineligible for any schedule by construction. Drawing
from it creates NO experimental exposure -- not because the draw is unreal, but
because no schedule may contain such a seed, so there is nothing to strike off.
Witnesses may be taken ONLY from it -- an ALLOWLIST, because the old denylist
silently drew from whatever nobody had thought to record, which is exactly what
happened to the ad hoc test seed 90000001.

Correspondingly there are TWO validation entry points, not one with a switch.
`validate_*_structure` asks whether a schedule is WELL FORMED and answers the
same way forever, so a completed run stays parseable, verifiable and analysable
as historical evidence. `validate_*_executable` asks whether it may RUN NOW, and
its answer changes the moment its seeds are spent. A `require_unspent=` keyword
would have made the second question defaultable, and a default that can be
switched off is exactly the gate-that-does-not-bind this workstream keeps
finding. Callers must name which question they are asking.

THE TWO STREAMS
---------------
`eval_runner.play_eval_game` derives two independent generators from one seed,
with colour-specific XOR masks, and the search generator must not be used for the
readout -- MCTS shares one `self.rng` across prior shuffle, PUCT tie-break and
readout, so drawing readout numbers from it would perturb every later search.
The masks are imported, never re-typed, so they cannot drift from that path.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Sequence, Tuple

from .twixtbot_g3_reference import SeededReferenceAgent, build_reference_agent, eval_config
from .twixtbot_g3_schedule import CONSUMED_SEEDS, REFERENCE_CHECKPOINTS

#: Every seed interval this workstream has reserved, consumed, or EXPOSED.
#: Half-open [start, end). These are seeds a schedule could use, so a draw from
#: one strikes it off even with no model and no game -- a witness may never be
#: taken from a seed inside these.
ACCOUNTED_SEED_INTERVALS = (
    (202608060, 202608124), (202608124, 202608188), (202608188, 202608988),
    (202608988, 202609388), (202609388, 202609788), (202609788, 202610188),
    (202611000, 202611400),          # twixtbot G3 declared block
    (202612000, 202612512),          # the E4 screen's reservation, see EXPOSED below
    (202613000, 202613064),          # L0 LARGER MATCH, reserved 2026-08-26 and
                                     # UNSPENT. 64 seeds: 8 openings x 2 colour
                                     # arms x 4 repetitions, all at mdPly 6.
                                     # Proved disjoint from every seed category
                                     # AND from every derived RNG stream before
                                     # reservation -- see the L0 card. Nothing has
                                     # been drawn from it; it is reserved, not
                                     # exposed.
    (202614000, 202614227),          # D1 SAME-POSITION INTERROGATION. 227 seeds,
                                     # one per retained position (plan 12.5), one
                                     # incumbent search/readout each; T1j has no
                                     # controlled seed at all, so duplicating a
                                     # T1j query consumes none of these.
                                     #
                                     # REGISTERED 2026-08-28, as part of the D1
                                     # EXECUTION authorization and not before.
                                     # Through preregistration and integration it
                                     # was reserved on paper and deliberately
                                     # absent from this tuple, so a block that was
                                     # never authorized would have cost nothing to
                                     # abandon; `d1_probe._check_seed_registration`
                                     # refused the run for exactly that reason.
                                     #
                                     # Disjointness was proved BEFORE reservation
                                     # against 3,429 prior seeds across every
                                     # category AND every derived RNG stream: 0
                                     # direct overlaps, 0 stream collisions, own
                                     # streams injective (1,135 distinct values
                                     # from 227 seeds x 5 derivations). The
                                     # enumeration was exhaustive, not sampled.
                                     #
                                     # Accounted, NOT exposed and NOT retired: a
                                     # reservation is not a draw. Those move only
                                     # once the run has actually drawn from it.
    (202615000, 202615221),          # D1 SAME-POSITION INTERROGATION, §14 RETRY.
                                     # 221 seeds, one per §13 Amendment 2 position,
                                     # bound POSITIONALLY: row i carries exactly
                                     # 202615000 + i, checked by `d1_probe`.
                                     #
                                     # REGISTERED 2026-09-08, as the seed-
                                     # PREPARATION step and NOT as an execution
                                     # authorization -- those are two separate
                                     # reviews, and this is only the first. At the
                                     # moment of this edit `D1_EXECUTION_AUTHORIZED`
                                     # is False and stays False; the run is still
                                     # refused, now by the gate alone. From §14 to
                                     # here the block was reserved on paper and
                                     # deliberately absent from this tuple, so a
                                     # block that was never authorized would have
                                     # cost nothing to abandon.
                                     #
                                     # RE-PROVED against the registries AS THEY
                                     # STAND -- which by then included BOTH spent
                                     # H1 blocks -- plus this paper reservation
                                     # itself. 4,109 prior seeds across six
                                     # categories, 20,545 prior values including
                                     # derivations: 0 direct overlaps, 0 derived-
                                     # stream collisions, own derivations
                                     # injective (1,105 = 221 x 5). Exhaustive,
                                     # not sampled. Eleven collision controls and
                                     # four gap-policy controls all rejected.
                                     #
                                     # 🔑 The candidate IS this paper reservation,
                                     # so it was excluded from the prior set BY
                                     # IDENTITY -- not by subtracting its seeds,
                                     # which would have excused any control equal
                                     # to an existing reservation. The first run
                                     # (v5) applied that exclusion to the overlap
                                     # check but NOT to the gap policy, and so
                                     # reported a distance-0 FAIL against itself;
                                     # v6 applies it to both and the nearest OTHER
                                     # boundary is 773 seeds away. Both runs are
                                     # kept in the evidence directory.
                                     #
                                     # Accounted, NOT exposed and NOT retired: a
                                     # reservation is not a draw.
    (202624000, 202624040),          # H3 PILOT. 40 seeds, one per game: 20
                                     # independently generated openings, each
                                     # played BOTH WAYS, bound POSITIONALLY
                                     # (row i = 202624000 + i).
                                     #
                                     # REGISTERED 2026-09-14 as the seed-PREPARATION
                                     # step and NOT an execution authorization. At
                                     # the moment of this edit
                                     # `H3_PILOT_EXECUTION_AUTHORIZED` is False and
                                     # stays False, and no pilot run is authorized.
                                     # Two separate reviews; this is the first.
                                     #
                                     # WHY A PILOT AT ALL. H2 is CLOSED: its
                                     # attempt 3 VOIDed on the deadline, and the
                                     # decisive fact was not the deadline -- 14 of
                                     # its 15 COMPLETE cells held ONE distinct game
                                     # in 46 repetitions, so 736 nominal games
                                     # carried ~61 distinct ones. The pilot tests
                                     # whether diversity from independently
                                     # generated POSITIONS works where repetition
                                     # did not, and it is deliberately small enough
                                     # that a pathological result costs two hours.
                                     #
                                     # RE-PROVED (collision proof v10) against the
                                     # registries as they stand, ALL THREE spent H2
                                     # blocks included: 6,357 prior seeds across six
                                     # categories, 31,785 prior values with
                                     # derivations, 0 direct overlaps, 0 derived-
                                     # stream collisions, own derivations injective
                                     # (200 = 40 x 5). Exhaustive, not sampled.
                                     # Seventeen collision controls and four gap-
                                     # policy controls all rejected.
                                     #
                                     # 🔑 Excluded from the prior set BY IDENTITY, in
                                     # the GAP check as well as the overlap check.
                                     # The gap floor is the candidate's OWN SIZE,
                                     # which is 40 here -- smaller than any earlier
                                     # round's because every earlier candidate was
                                     # larger. The policy is unchanged and the
                                     # NEAREST ACTUAL BOUNDARY IS 1,264 away, with
                                     # nothing above; the distance is reported so a
                                     # narrow choice could not hide behind a small
                                     # threshold.
                                     #
                                     # Accounted, NOT exposed and NOT retired: a
                                     # reservation is not a draw.
    (202622000, 202622736),          # H2 ATTEMPT 3 -- the match H2 has still not
                                     # had. 736 seeds, one per game, bound
                                     # POSITIONALLY (row i = 202622000 + i).
                                     #
                                     # REGISTERED 2026-09-12 as the seed-PREPARATION
                                     # step, NOT an execution authorization: at the
                                     # moment of this edit `H2_EXECUTION_AUTHORIZED`
                                     # is False and stays False, and no match is
                                     # authorized. Two separate reviews; this is the
                                     # first.
                                     #
                                     # WHY A THIRD BLOCK. Attempt 1 VOIDed at task 0
                                     # (the builder refused H2's own readout) and was
                                     # retired whole with 0 exposed. Attempt 2 was
                                     # never authorized to run at all: on 2026-09-12
                                     # an injected-defect control deleted
                                     # `check_gate()` from `run_h2` and played 383 of
                                     # its 736 games, so it is ACCOUNTED 736 /
                                     # EXPOSED 383 / RETIRED WHOLE and its 383 games
                                     # are INCIDENT EVIDENCE, not an H2 verdict.
                                     #
                                     # RE-PROVED (collision proof v9) against the
                                     # registries as they stand, BOTH spent blocks
                                     # included: 6,317 prior seeds across six
                                     # categories, 31,585 prior values with
                                     # derivations, 0 direct overlaps, 0 derived-
                                     # stream collisions, own derivations injective
                                     # (3,680 = 736 x 5). Exhaustive, not sampled.
                                     # Seventeen collision controls and four gap-
                                     # policy controls all rejected, both spent H2
                                     # blocks among them.
                                     #
                                     # 🔑 Excluded from the prior set BY IDENTITY, in
                                     # the GAP check as well as the overlap check. The
                                     # gap floor is the candidate's OWN SIZE (736);
                                     # nearest other boundary 1,264, nothing above.
                                     #
                                     # Accounted, NOT exposed and NOT retired: a
                                     # reservation is not a draw.
    (202620000, 202620736),          # H2 ATTEMPT 2 -- THE RETRY against the
                                     # REPAIRED builder. 736 seeds, one per game,
                                     # bound POSITIONALLY (row i = 202620000 + i).
                                     #
                                     # REGISTERED 2026-09-11 as the seed-PREPARATION
                                     # step, NOT an execution authorization: at the
                                     # moment of this edit `H2_EXECUTION_AUTHORIZED`
                                     # is False and stays False.
                                     #
                                     # WHY A FRESH BLOCK. Attempt 1's block below was
                                     # registered and then RETIRED WHOLE the same day:
                                     # its match VOIDed at task 0 because
                                     # `build_reference_agent` refused H2's own
                                     # readout. No seed was drawn (exposed 0), and the
                                     # retirement was REVIEWED AND KEPT -- fresh seeds
                                     # cost less than a retrospective exception.
                                     #
                                     # RE-PROVED against the registries as they stand,
                                     # attempt 1's spent block INCLUDED, plus this
                                     # reservation itself: 5,581 prior seeds across six
                                     # categories, 27,905 prior values with
                                     # derivations, 0 direct overlaps, 0 derived-stream
                                     # collisions, own derivations injective
                                     # (3,680 = 736 x 5). Exhaustive, not sampled.
                                     # Fourteen collision controls and four gap-policy
                                     # controls all rejected, attempt 1's block among
                                     # them.
                                     #
                                     # 🔑 Excluded from the prior set BY IDENTITY, in
                                     # the GAP check as well as the overlap check. The
                                     # gap floor is the candidate's OWN SIZE (736);
                                     # nearest other boundary 1,264.
                                     #
                                     # Accounted, NOT exposed and NOT retired.
    (202618000, 202618736),          # H2 DETERMINISTIC-READOUT HEAD-TO-HEAD. 736
                                     # seeds, one per game: 8 openings x 2 colour
                                     # arms x 46 repetitions at mdPly 6, bound
                                     # POSITIONALLY (row i carries 202618000 + i).
                                     #
                                     # REGISTERED 2026-09-11, as the seed-
                                     # PREPARATION step and NOT as an execution
                                     # authorization -- two separate reviews, and
                                     # this is only the first. At the moment of
                                     # this edit `H2_EXECUTION_AUTHORIZED` is
                                     # False and stays False; the run is refused
                                     # by the gate alone. From the card to here
                                     # the block was reserved on paper and
                                     # deliberately absent from this tuple.
                                     #
                                     # RE-PROVED against the registries AS THEY
                                     # STAND -- which now include EVERY earlier
                                     # block, D1's §14 among them, spent and
                                     # retired on 2026-09-08 -- plus this paper
                                     # reservation itself. 4,845 prior seeds
                                     # across six categories, 24,225 prior values
                                     # including derivations: 0 direct overlaps,
                                     # 0 derived-stream collisions, own
                                     # derivations injective (3,680 = 736 x 5).
                                     # Exhaustive, not sampled. Thirteen collision
                                     # controls and four gap-policy controls all
                                     # rejected.
                                     #
                                     # 🔑 The candidate IS this paper reservation,
                                     # so it was excluded from the prior set BY
                                     # IDENTITY -- in the GAP check as well as the
                                     # overlap check, which is the correction the
                                     # D1 round had to make mid-proof.
                                     #
                                     # 🔑 THE GAP FLOOR IS THE CANDIDATE'S OWN
                                     # SIZE (736, not the old fixed 224), so the
                                     # gap can never be smaller than the block it
                                     # protects. Nearest other boundary: 776.
                                     #
                                     # Accounted, NOT exposed and NOT retired: a
                                     # reservation is not a draw.
    (202616000, 202616224),          # H1 HEAD-TO-HEAD VIABILITY SCREEN. 224
                                     # seeds, one per game: 8 openings x 2 colour
                                     # arms x 14 repetitions at mdPly 6.
                                     #
                                     # REGISTERED 2026-09-04, as the seed-
                                     # preparation step of the H1 authorization
                                     # and not before. Through the card, the plan
                                     # and the runner it was reserved on paper and
                                     # deliberately absent from this tuple, so a
                                     # block that was never authorized would have
                                     # cost nothing to abandon;
                                     # `h1_viability_runner.check_seed_registration`
                                     # refused the run for exactly that reason.
                                     #
                                     # 🔴 DISJOINTNESS WAS RE-PROVED AGAINST THE
                                     # REGISTRIES AS THEY STOOD, PLUS
                                     # `d1_selection.SEED_INTERVAL` AS AN EXPLICIT
                                     # TERM. That D1 block is reserved on paper
                                     # and is in NO registry by design, so a
                                     # registry-only enumeration cannot see it --
                                     # reserved-on-paper is still TAKEN, and
                                     # without that term an overlapping block
                                     # would have passed every check here.
                                     #
                                     # 3,661 prior seeds across six categories: 0
                                     # direct overlaps, 0 derived-stream
                                     # collisions against 18,305 prior values, own
                                     # derivations injective (1,120 = 224 x 5,
                                     # both colours). Exhaustive, not sampled.
                                     # Six negative controls all rejected,
                                     # including D1's paper block and a
                                     # stream-only collision with NO direct
                                     # overlap.
                                     #
                                     # Accounted, NOT exposed and NOT retired.
                                     # Registering it does NOT open the gate:
                                     # H1_EXECUTION_AUTHORIZED is a separate
                                     # constant and remains False.
    (202617000, 202617224),          # H1 ATTEMPT 2, registered 2026-09-07 as
                                     # ACCOUNTED ONLY: 0 exposed, 0 retired --
                                     # a reservation is not a draw. Attempt 1
                                     # ([202616000, 202616224) above) VOIDed at
                                     # game 60 and retired WHOLE; this is the
                                     # fresh interval, collision-proved twice
                                     # (retry-prep v3b; seed-registration v4,
                                     # the spent block named by ITS OWN name)
                                     # against every category incl. D1's paper
                                     # block and the spent block, 0 direct,
                                     # 0 derived-stream, with a 776-seed gap.
                                     # Registering it does NOT open the gate:
                                     # H1_EXECUTION_AUTHORIZED stays False.
    (202626000, 202626592),          # 🔴 H3 FULL STUDY -- the match itself. 592
                                     # seeds, ONE PER GAME, bound POSITIONALLY
                                     # (row i = 202626000 + i) across four
                                     # contiguous 148-seed segment quarters:
                                     #   segment 0  [202626000, 202626148)
                                     #   segment 1  [202626148, 202626296)
                                     #   segment 2  [202626296, 202626444)
                                     #   segment 3  [202626444, 202626592)
                                     # 296 colour-reversed pairs, both arms
                                     # adjacent, over the population FROZEN
                                     # 2026-09-17 and PINNED 2026-09-18
                                     # (opening_set_digest 35932b3f...f772e46).
                                     #
                                     # REGISTERED 2026-09-18 as the seed-PREPARATION
                                     # step and NOT an execution authorization. At
                                     # the moment of this edit
                                     # `H3_STUDY_EXECUTION_AUTHORIZED` is False and
                                     # stays False, and no segment is authorized.
                                     # Two separate reviews; this is the first.
                                     #
                                     # PROVED by collision proof v14 against the
                                     # registries as they stand: 302,357 prior
                                     # seeds across eleven categories, 1,511,785
                                     # prior values with derivations, 0 direct
                                     # overlaps, 0 derived-stream collisions, own
                                     # derivations injective (2,960 = 592 x 5).
                                     # Exhaustive, not sampled. Fifteen negative
                                     # controls all rejected.
                                     #
                                     # 🔴 THE FOUR GENERATION RANGES ARE IN NO
                                     # REGISTRY and v14 adds them by hand -- three
                                     # retired, plus the LIVE range that produced
                                     # the frozen population. A registry-only
                                     # enumeration would call an overlapping block
                                     # clean, and three of the fifteen controls sit
                                     # inside those ranges precisely to prove it
                                     # does not.
                                     #
                                     # 🔑 Excluded from the prior set BY IDENTITY,
                                     # in the GAP check as well as the overlap
                                     # check. The gap floor is the candidate's OWN
                                     # SIZE, 592 here; the NEAREST ACTUAL BOUNDARY
                                     # IS 1,960 away, with nothing above. The
                                     # distance is reported so a narrow choice
                                     # could not hide behind a small threshold.
                                     #
                                     # Accounted, NOT exposed and NOT retired: a
                                     # reservation is not a draw.
    (202628000, 202628148),          # 🔴 H3 SEGMENT 0's REPLACEMENT BLOCK. 148
                                     # seeds, one per game, bound POSITIONALLY
                                     # (row i = 202628000 + i within segment 0).
                                     #
                                     # WHY A REPLACEMENT. Segment 0's original
                                     # quarter [202626000, 202626148) RETIRED
                                     # WHOLE on the VOID of 2026-09-18 -- exposed
                                     # 0, retired all the same, because the
                                     # authorization said the quarter retires ON
                                     # START. No strength information was
                                     # produced, so replacing these seeds
                                     # introduces no outcome-based selection.
                                     #
                                     # 🔑 AND THE BLOCK IS NOW FOUR, NOT ONE. The
                                     # study used a single 592-seed interval whose
                                     # four quarters were ALL required unspent to
                                     # construct ANY segment, so retiring segment
                                     # 0's blocked segments 1-3 too -- the
                                     # opposite of what segmenting is for.
                                     # Segments 1-3 keep [202626148, 202626592);
                                     # this replaces segment 0 alone.
                                     #
                                     # PROVED by collision proof v15: 302,505
                                     # prior seeds across thirteen categories,
                                     # 1,514,745 prior values with derivations, 0
                                     # direct overlaps, 0 derived-stream
                                     # collisions, own derivations injective
                                     # (740 = 148 x 5). Exhaustive, not sampled.
                                     # EIGHTEEN negative controls all rejected,
                                     # segment 0's own retired quarter and the
                                     # whole original 592-seed block among them.
                                     #
                                     # 🔑 Excluded BY IDENTITY in the GAP check as
                                     # well as the overlap check. Gap floor is the
                                     # candidate's OWN SIZE, 148; nearest actual
                                     # boundary is 1,408 away.
                                     #
                                     # Accounted, NOT exposed and NOT retired.
                                     # Registering it does not open the gate:
                                     # H3_STUDY_EXECUTION_AUTHORIZED stays False.
    # 🔴 H4 -- FIVE BLOCKS, registered 2026-09-24 (step 4b) by the BOUND step-4
    # card §2 (`2026-09-24-t1j-h4-step4-seed-card.md`). One seed per game; pair k
    # gets lo+2k (Arm A) and lo+2k+1 (Arm B). Registered ONLY after collision proof
    # v16 (amended, card amendment 6) was CLEAN against the PRE-registration
    # registries, 14/14 controls rejected
    # (evidence/2026-09-24-t1j-h4-seed-registration/02b_*). Its first pre-run was
    # NOT CLEAN on a self-referential literal term (02a, preserved).
    #
    # 🔴 USE IS NOT RECORDED HERE UNTIL H4 CLOSES (card §2.1): this file is hashed
    # into H4's `code`, so an EXPOSED/RETIRED edit between runs would void the
    # study manifests. Until then each run's occupied create-only evidence
    # directory is its use record; ONE closing edit derives exposure/retirement.
    # Accounted, NOT exposed and NOT retired. No gate is opened by this.
    (202632000, 202632032),          # H4 PILOT            -- 16 pairs, 32 seeds
    (202634000, 202634148),          # H4 STUDY SEGMENT 0  -- 74 pairs, 148 seeds
    (202636000, 202636148),          # H4 STUDY SEGMENT 1  -- 74 pairs, 148 seeds
    (202638000, 202638148),          # H4 STUDY SEGMENT 2  -- 74 pairs, 148 seeds
    (202640000, 202640148),          # H4 STUDY SEGMENT 3  -- 74 pairs, 148 seeds
)

#: EXPERIMENTAL EXPOSURE: seeds drawn from OUTSIDE the permanently unschedulable
#: test namespace below. Half-open [start, end).
#:
#: Drawing alone is not what this records, and saying so would contradict the test
#: band: those are drawn from constantly and never appear here. What makes a draw
#: an exposure is that it happened to a seed A SCHEDULE COULD HAVE USED, which
#: strikes that seed off. A draw inside the test namespace strikes nothing off,
#: because nothing there was ever available to schedule.
EXPOSED_SEED_INTERVALS = (
    (202624000, 202624040),          # H3 PILOT, DRAWN BY THE ONE AUTHORIZED RUN on
                                     # 2026-09-15 (00:47:15Z -> 01:14:44Z, 27m29s).
                                     # ALL 40 seeds, bound POSITIONALLY:
                                     # 202624000..202624039.
                                     #
                                     # THE RUN COMPLETED, 40/40, wrapper exit 0.
                                     # Every seed carries a COMPLETED game: 40
                                     # task_result records, 40 opening_bound, 40
                                     # transcripts, 40 task_start each with a
                                     # matching task_done. There is no
                                     # partially-drawn seed to judge, which is the
                                     # question attempts 2 and 3 of H2 each had to
                                     # answer differently.
                                     #
                                     # 🔴 DERIVED, NOT READ OFF THE RECORDS. H2's
                                     # `task_result` carried `seed`; H3's does not --
                                     # `_play_one` returns it and the run body's
                                     # projection drops it. The binding is sound and
                                     # pinned: the durable header carries the SEEDED
                                     # full-field task digest
                                     # aa527cc9a1a7b1e657911171c63f19fc006909dd64518bd96de3ce4ddfabfba9,
                                     # which matches SEEDED_TASK_DIGEST, and
                                     # `build_tasks` assigns row i -> lo + i. But it
                                     # is a DERIVATION from a pin, where H2's was a
                                     # value in the record. Recorded here so the
                                     # accounting states its own provenance.
    (202622000, 202622693),          # H2 ATTEMPT 3, DRAWN BY THE ONE AUTHORIZED
                                     # RUN on 2026-09-13. 693 seeds, bound
                                     # POSITIONALLY: 202622000..202622692.
                                     #
                                     # 692 of them carry a COMPLETED game -- a
                                     # task_result, an opening_bound, a transcript
                                     # and ply records each, indices 0..691.
                                     #
                                     # 🔑 THE 693rd IS CLAIMED AS DRAWN, and that is
                                     # a DIFFERENT judgement from attempt 2's. There
                                     # the uncertain seed emitted a trace
                                     # `task_start` and NOTHING else, so it was not
                                     # claimed. Here task 692 emitted `task_start`
                                     # AND the run aborted inside it -- "[move]
                                     # h2match-692-strong6-o8_contact-t1j_black-r2
                                     # ply 9" -- so an agent was built on seed
                                     # 202622692 and nine plies were played with it.
                                     # Its records did not survive, because ply
                                     # records are persisted per COMPLETED game; the
                                     # draw did. An unrecorded draw is still a draw.
    (202620000, 202620383),          # H2 ATTEMPT 2's block, DRAWN BY AN
                                     # UNAUTHORIZED RUN on 2026-09-12: an
                                     # injected-defect control deleted the gate
                                     # check from the public entry, and with the
                                     # production seam now wired the entry compiled
                                     # the helper, loaded the model and PLAYED 383
                                     # REAL GAMES before it was killed.
                                     #
                                     # 383 CONFIRMED, COUNTED FROM THE RECORDS: 383
                                     # task_result records, 383 opening_bound, 383
                                     # transcripts and 14,708 ply records, seeds
                                     # 202620000..202620382 inclusive.
                                     #
                                     # ⚠ SEED 202620383 IS UNCERTAIN AND DELIBERATELY
                                     # NOT CLAIMED HERE. Task 383 has a `task_start`
                                     # trace line and NOTHING else -- no
                                     # opening_bound, no ply, no result -- which by
                                     # H1's rule means its agent was never built and
                                     # its seed never drawn. But this run was KILLED
                                     # rather than aborted, so construction could
                                     # have been in flight. Exposure records what the
                                     # evidence SHOWS; the whole-block retirement
                                     # below covers what it cannot rule out.
    (202615000, 202615221),          # D1 SAME-POSITION INTERROGATION, §14 RETRY,
                                     # run once 2026-09-08 and COMPLETED: 221 of
                                     # 221 positions, 1,105 of 1,105 queries,
                                     # trace verdict OK.
                                     #
                                     # ALL 221 DRAWN, COUNTED FROM THE RECORD,
                                     # not asserted from the design: every one of
                                     # the 221 position entries carries an
                                     # incumbent readout, and the readout is what
                                     # draws the seed. The trace agrees
                                     # independently (`seeds_drawn: 221` at
                                     # run_end). Unlike the 2026-08-28 VOID, which
                                     # was retired with NO exposure claim because
                                     # claiming 227 draws would have asserted 226
                                     # that may never have happened, here the
                                     # record survives, so the count is exact.
    (202617000, 202617224),          # H1 ATTEMPT 2, run once 2026-09-07 and
                                     # COMPLETED: 224 of 224 games, every task
                                     # built our reference agent (224 task_result
                                     # records, >= 1 reference ply each), so ALL
                                     # 224 seeds were DRAWN. Counted from the
                                     # records, not asserted from the design.
    (202616000, 202616060),          # THE H1 MATCH, run once 2026-09-05 and
                                     # VOIDED at game 60 of 224. These 60 seeds
                                     # built real SeededReferenceAgents and drove
                                     # real generators: 60 games were played to a
                                     # terminal state and recorded.
                                     #
                                     # 🔑 SIXTY, NOT SIXTY-ONE. Game 60 started
                                     # and its opening was bound, but the arm is
                                     # `t1j_red` and T1j was to move at ply 6; its
                                     # query failed there, so our reference agent
                                     # -- which plays black -- was never
                                     # constructed and seed 202616060 was never
                                     # drawn. The task has task_start and
                                     # opening_bound records and ZERO ply records,
                                     # which is what makes that checkable.
                                     #
                                     # D1's VOID could not be counted this way and
                                     # was retired without an exposure claim,
                                     # because claiming 227 draws would have
                                     # asserted 226 that may never have happened.
                                     # Here the record survives, so the count is
                                     # exact rather than bounded.
    (202612128, 202612136),          # THE E4 CANONICAL SCREEN, strong endpoint,
    (202612144, 202612160),          # tasks 000-007, and the weak endpoint, tasks
                                     # 016-031: 24 tasks PLAYED on 2026-08-26 from
                                     # a8b3994. Their seeds drove real generators.
                                     # Tasks 008-015 (202612136..202612143) were
                                     # SKIPPED by the recorded early stop and were
                                     # never drawn from -- they are undrawn, and
                                     # are retired below rather than claimed here.
    (202612000, 202612032),          # E4 preflight attempt 3 witnesses
    (90002000, 90002004),            # E4 integration qualification ATTEMPT 2,
                                     # 2026-08-26: the corrective run, same four
                                     # tasks, same drawing. Spent.
    (90001000, 90001004),            # E4 INTEGRATION qualification, 2026-08-25:
                                     # four synthetic tasks each built a real
                                     # SeededReferenceAgent and drew from both
                                     # generators. Spent, model or no model.
    (202613000, 202613064),          # THE L0 CANONICAL 64-GAME MATCH, executed
                                     # once from 9805c19 on 2026-08-27. ALL 64
                                     # tasks played -- L0 has no early stop -- so
                                     # every seed in the block drove real
                                     # generators. Drawn, not merely reserved.
    (90000001, 90000002),            # THE OLD `SYNTHETIC` TEST SEED, which was
                                     # SCHEDULABLE. Drawn from TWICE, both draws
                                     # preserved:
                                     #  1) 2026-08-25-t1j-e4-preflight-attempt4/
                                     #     06_endpoint_screen_plan.json, at
                                     #     seed_accounting.witness_demonstration
                                     #     -- an rng_witness frozen into the
                                     #     canonical plan, four values per stream.
                                     #  2) 2026-08-25-t1j-e4-harness-qualification/
                                     #     04_qualify.py.txt:26,50,70 binds it to
                                     #     the one real agent call, and
                                     #     02_qualification.txt:18-28 is that call
                                     #     RUNNING: completed, one move (14,13),
                                     #     "search RNG advanced" + "readout RNG
                                     #     advanced".
                                     # It was absent from this registry anyway, and
                                     # a test asserted it must stay usable -- the
                                     # exact reuse this registry exists to prevent.
                                     # Recorded 2026-08-26.
    (202628000, 202628148),          # 🔴 H3 SEGMENT 0 -- RAN AND COMPLETED
                                     # 2026-09-18. 148/148 games, wrapper exit 0,
                                     # verdict OK, 2.12 h of a 3.00 h cap. EVERY
                                     # one of the 148 seeds was DRAWN, read from
                                     # the records themselves (each task_result
                                     # carries its own seed), not derived from
                                     # the plan.
                                     #
                                     # RETIRED WHOLE under the one-shot rule.
                                     # Segments 1-3 keep [202626148, 202626592),
                                     # untouched -- which is the point of the
                                     # 2026-09-18 isolation repair.
    (202626148, 202626296),          # 🔴 H3 SEGMENT 1 -- RAN AND COMPLETED
                                     # 2026-09-19. 148/148 games, wrapper exit 0,
                                     # verdict OK, 1.86 h of a 3.00 h cap. EVERY
                                     # one of the 148 seeds was DRAWN, read from
                                     # the records themselves (each task_result
                                     # carries its own seed, 148 distinct across
                                     # [202626148, 202626295]), not derived from
                                     # the plan.
                                     #
                                     # RETIRED WHOLE under the one-shot rule.
                                     # Segments 2-3 keep [202626296, 202626592),
                                     # untouched.
    (202626296, 202626444),          # 🔴 H3 SEGMENT 2 -- RAN AND COMPLETED
                                     # 2026-09-19. 148/148 games, wrapper exit 0,
                                     # verdict OK, 1.84 h of a 3.00 h cap. EVERY
                                     # one of the 148 seeds was DRAWN, read from
                                     # the records themselves (each task_result
                                     # carries its own seed, 148 distinct across
                                     # [202626296, 202626443]), not derived from
                                     # the plan.
                                     #
                                     # RETIRED WHOLE under the one-shot rule.
                                     # Segment 3 keeps [202626444, 202626592).
    (202626444, 202626592),          # 🔴 H3 SEGMENT 3 -- RAN AND COMPLETED
                                     # 2026-09-20. 148/148 games, wrapper exit 0,
                                     # verdict OK, 2.06 h of a 3.00 h cap. EVERY
                                     # one of the 148 seeds was DRAWN, read from
                                     # the records themselves (each task_result
                                     # carries its own seed, 148 distinct across
                                     # [202626444, 202626591]), not derived from
                                     # the plan.
                                     #
                                     # RETIRED WHOLE under the one-shot rule.
                                     # THE LAST SEGMENT: all four quarters of the
                                     # H3 full study are now spent.
)

#: Seeds RESERVED PERMANENTLY FOR TESTS. Unit tests must draw from something, and
#: a draw from a schedulable seed strikes that seed off. Naming a fresh
#: "synthetic" seed each time is what put 90000001 above: it was schedulable, it
#: was drawn from repeatedly, and nobody recorded it.
#:
#: So the tests draw from a band that is INELIGIBLE FOR SCHEDULING BY
#: CONSTRUCTION. Draws here are just as real, and create NO experimental exposure,
#: because no schedule may contain one -- `validate_schedule_executable` refuses
#: them, and they lie outside the screen's reserved block, so the two barriers are
#: independent. They are therefore NOT listed as exposed: there is nothing to
#: strike off, not a draw to hide.
TEST_ONLY_SEED_INTERVALS = (
    (90009000, 90009100),
)

#: Seeds RETIRED ADMINISTRATIVELY. Not a claim that anything was drawn: a claim
#: that the block may not be used again. The canonical screen was a preregistered
#: ONE-SHOT schedule and it completed. Replaying its 8 undrawn seeds would select
#: exactly the tasks the early stop declined to play -- a result chosen after
#: seeing the first 24, which is selection bias however clean the RNG is. So the
#: WHOLE block retires together, drawn and undrawn alike.
RETIRED_SEED_INTERVALS = (
    (202624000, 202624040),          # H3 PILOT, RETIRED WHOLE 2026-09-15. All 40
                                     # were drawn (see EXPOSED above), so whole-block
                                     # retirement and exposure coincide here for the
                                     # first time in this programme -- the pilot is
                                     # the first one-shot schedule to COMPLETE.
                                     #
                                     # The one-shot rule applies regardless: a
                                     # schedule that has run may not be re-run, and
                                     # selecting any part of it again would be
                                     # choosing tasks after seeing the result. A
                                     # second pilot needs a FRESH block with its own
                                     # collision re-proof.
    (202622000, 202622736),          # H2 ATTEMPT 3, RETIRED WHOLE 2026-09-13.
                                     # 693 of its seeds were drawn by the single
                                     # authorized run (see EXPOSED above) and the
                                     # remaining 43 go with them: a one-shot
                                     # schedule was STARTED and did not complete.
                                     #
                                     # THE RUN VOIDED ON ITS OWN DEADLINE at game
                                     # 692 of 736 -- 28,800 s exceeded, exit 3, the
                                     # durable trace's `run_end` reading
                                     # verdict VOID / games_completed 692. The gate
                                     # was restored by the wrapper and every gate
                                     # reads False.
                                     #
                                     # ⚠ NO STRENGTH CONCLUSION IS DRAWN FROM THE
                                     # 692 GAMES. The design requires all 736 and
                                     # the frozen rules produce no verdict from a
                                     # VOID: no report was written. They are
                                     # FAILURE EVIDENCE -- what the authorized
                                     # launch path did -- and nothing else.
    (202620000, 202620736),          # H2 ATTEMPT 2, RETIRED WHOLE 2026-09-12.
                                     # 383 of its seeds were drawn by an
                                     # UNAUTHORIZED run (see EXPOSED above) and the
                                     # remaining 353 go with them: a one-shot
                                     # schedule was started and did not complete.
                                     # This also covers the one seed whose use
                                     # cannot be ruled out, 202620383.
                                     #
                                     # 🔴 THE BLOCK WAS REGISTERED THE DAY BEFORE AND
                                     # SPENT BY MY OWN TEST HARNESS, not by an
                                     # authorized match. The games it played are
                                     # INCIDENT evidence and yield no H2 verdict.
                                     # A third block may not be reserved until the
                                     # harness containment is verified -- otherwise
                                     # it is just another resource for the same
                                     # failure to consume.
    (202618000, 202618736),          # H2, retired WHOLE 2026-09-11 after the single
                                     # authorized match VOIDed at task 0, ply 7:
                                     # `build_reference_agent` refuses any config
                                     # that is not the frozen research one, and
                                     # H2 IS a one-field change to it. No agent was
                                     # constructed, no move played, no seed drawn.
                                     #
                                     # NOT EXPOSED: zero ply records exist, so
                                     # claiming any draw would assert something
                                     # that did not happen -- the same evidence
                                     # test H1 used at its game 60.
                                     #
                                     # RETIRED WHOLE on the frozen one-shot rule:
                                     # a preregistered schedule was STARTED and did
                                     # not complete. ⚠ The rule's stated reason --
                                     # that replaying part of it would select after
                                     # seeing where it failed -- arguably does not
                                     # bite here, since nothing was played and
                                     # nothing learned about any game. The rule is
                                     # applied anyway: relaxing one the moment it
                                     # costs something is how rules stop binding.
                                     # A FUTURE H2 NEEDS A FRESH INTERVAL.
    (202615000, 202615221),          # D1 §14, retired WHOLE 2026-09-08: a
                                     # preregistered ONE-SHOT schedule that
                                     # COMPLETED (analysis outcome NO_GO). L0's
                                     # rule -- replaying any part of it would be
                                     # selection after seeing the result. Here
                                     # every seed was also drawn, so "whole" and
                                     # "drawn" coincide; the rule is the reason
                                     # regardless.
    (202617000, 202617224),          # H1 ATTEMPT 2, retired WHOLE 2026-09-07:
                                     # a preregistered ONE-SHOT schedule that
                                     # COMPLETED (verdict INCONCLUSIVE). L0's
                                     # rule: replaying any part of it would be
                                     # selection after seeing the result.
    (202616000, 202616224),          # THE H1 BLOCK, retired WHOLE 2026-09-05
                                     # after the single authorized match VOIDED
                                     # at game 60 of 224 (34m02s of a 180-minute
                                     # window; the deadline was not the cause).
                                     #
                                     # CAUSE, recorded because this VOID could
                                     # name it: T1j MUTATED THE HOST PREFERENCE
                                     # STORE. `prefs_ok=false, failures=1` on the
                                     # ply-6 query of h1match-060, which is a
                                     # frozen abort rule -- "any postcondition
                                     # failure: a Window/Frame, a non-headless
                                     # jvm, a mutated host preference store, or an
                                     # unauthorized reflective access".
                                     #
                                     # RETIRED WHOLE, drawn and undrawn alike, on
                                     # the rule the canonical screen's and D1's
                                     # blocks retired under: a preregistered
                                     # ONE-SHOT schedule was started and did not
                                     # complete, so replaying any part of it would
                                     # select games after seeing where it failed.
                                     # The 164 undrawn seeds go with it.
                                     #
                                     # A future H1 needs a FRESH interval.
    (202612128, 202612160),          # the canonical screen's 32-seed block
    (202613000, 202613064),          # the L0 match's 64-seed block. Its
                                     # preregistered ONE-SHOT schedule completed
                                     # on 2026-08-27, so the block retires whole.
                                     # Unlike the screen's, every seed here was
                                     # also drawn: L0 played all 64.
    (202614000, 202614227),          # THE D1 BLOCK, retired 2026-08-28 after the
                                     # single authorized run VOIDED at 3m18s
                                     # (exit 3, no record written).
                                     #
                                     # AT LEAST ONE seed was drawn: the VOID came
                                     # from _probe_position, which runs after that
                                     # position's incumbent readout, and building
                                     # that agent constructs both generators and
                                     # runs a 400-simulation search.
                                     #
                                     # HOW MANY were drawn is UNDETERMINED. The
                                     # record is written once at the end, so a
                                     # VOID leaves no per-position trace, and the
                                     # VOID message names the depth and invocation
                                     # but not the position. So this is RETIRED
                                     # and deliberately NOT listed as EXPOSED:
                                     # claiming 227 draws would assert 226 that
                                     # may never have happened, which is the
                                     # overstatement these two lists are kept
                                     # apart to prevent.
                                     #
                                     # Retired WHOLE, drawn and undrawn alike --
                                     # the same rule the canonical screen's block
                                     # retired under. A preregistered one-shot
                                     # schedule was started and did not complete;
                                     # replaying any part of it would select
                                     # positions after seeing where it failed.
                                     # A future D1 needs a FRESH interval.
    (202626000, 202626148),          # 🔴 H3 SEGMENT 0's QUARTER, RETIRED WHOLE
                                     # 2026-09-18 on a VOID. The single
                                     # authorized attempt died in ONE SECOND at
                                     # its first durable write:
                                     # `_run_segment_unguarded` opens the trace
                                     # with O_EXCL and never creates the parent
                                     # directory, which did not exist.
                                     #
                                     # EXPOSED 0. No model was loaded, no JVM
                                     # started, no agent seeded, no game began --
                                     # the failure is strictly before the first
                                     # game. The same accounting H2 attempt 1
                                     # took when it VOIDed at task 0.
                                     #
                                     # RETIRED ALL THE SAME, because the
                                     # authorization said so in advance: the
                                     # quarter retires whole ON START, not on
                                     # success. A future segment 0 needs a FRESH
                                     # quarter with its own collision re-proof.
                                     # Segments 1-3 keep [202626148, 202626592).
    (202628000, 202628148),          # 🔴 H3 SEGMENT 0 -- RAN AND COMPLETED
                                     # 2026-09-18. 148/148 games, wrapper exit 0,
                                     # verdict OK, 2.12 h of a 3.00 h cap. EVERY
                                     # one of the 148 seeds was DRAWN, read from
                                     # the records themselves (each task_result
                                     # carries its own seed), not derived from
                                     # the plan.
                                     #
                                     # RETIRED WHOLE under the one-shot rule.
                                     # Segments 1-3 keep [202626148, 202626592),
                                     # untouched -- which is the point of the
                                     # 2026-09-18 isolation repair.
    (202626148, 202626296),          # 🔴 H3 SEGMENT 1 -- RAN AND COMPLETED
                                     # 2026-09-19. 148/148 games, wrapper exit 0,
                                     # verdict OK, 1.86 h of a 3.00 h cap. EVERY
                                     # one of the 148 seeds was DRAWN, read from
                                     # the records themselves (each task_result
                                     # carries its own seed, 148 distinct across
                                     # [202626148, 202626295]), not derived from
                                     # the plan.
                                     #
                                     # RETIRED WHOLE under the one-shot rule.
                                     # Segments 2-3 keep [202626296, 202626592),
                                     # untouched.
    (202626296, 202626444),          # 🔴 H3 SEGMENT 2 -- RAN AND COMPLETED
                                     # 2026-09-19. 148/148 games, wrapper exit 0,
                                     # verdict OK, 1.84 h of a 3.00 h cap. EVERY
                                     # one of the 148 seeds was DRAWN, read from
                                     # the records themselves (each task_result
                                     # carries its own seed, 148 distinct across
                                     # [202626296, 202626443]), not derived from
                                     # the plan.
                                     #
                                     # RETIRED WHOLE under the one-shot rule.
                                     # Segment 3 keeps [202626444, 202626592).
    (202626444, 202626592),          # 🔴 H3 SEGMENT 3 -- RAN AND COMPLETED
                                     # 2026-09-20. 148/148 games, wrapper exit 0,
                                     # verdict OK, 2.06 h of a 3.00 h cap. EVERY
                                     # one of the 148 seeds was DRAWN, read from
                                     # the records themselves (each task_result
                                     # carries its own seed, 148 distinct across
                                     # [202626444, 202626591]), not derived from
                                     # the plan.
                                     #
                                     # RETIRED WHOLE under the one-shot rule.
                                     # THE LAST SEGMENT: all four quarters of the
                                     # H3 full study are now spent.
)


def seed_is_accounted(seed: int) -> bool:
    return any(lo <= int(seed) < hi for lo, hi in ACCOUNTED_SEED_INTERVALS)


def seed_is_exposed(seed: int) -> bool:
    """Was this SCHEDULABLE seed drawn from, and so struck off? See the registry.

    Not simply "was it drawn from": test-namespace seeds are drawn from constantly
    and are never exposed, because they were never available to schedule.
    """
    return any(lo <= int(seed) < hi for lo, hi in EXPOSED_SEED_INTERVALS)


def seed_is_test_only(seed: int) -> bool:
    """Reserved for tests: drawable without limit, schedulable never.

    Drawing here is a real draw. It creates no exposure only because nothing in
    this band could ever have entered a schedule.
    """
    return any(lo <= int(seed) < hi for lo, hi in TEST_ONLY_SEED_INTERVALS)


def seed_is_retired(seed: int) -> bool:
    """Is this seed administratively withdrawn? A rule about the future."""
    return any(lo <= int(seed) < hi for lo, hi in RETIRED_SEED_INTERVALS)


def seed_is_unavailable(seed: int) -> bool:
    """Exposed OR retired: either way it may never be scheduled again."""
    return seed_is_exposed(seed) or seed_is_retired(seed)


def seed_status(seed: int) -> Dict[str, bool]:
    """The two registries reported SEPARATELY, never merged into one word."""
    return {"exposed": seed_is_exposed(seed), "retired": seed_is_retired(seed),
            "accounted": seed_is_accounted(seed), "test_only": seed_is_test_only(seed)}


#: Exactly the fields `build_reference_agent` reads off a task.
REQUIRED_TASK_FIELDS = ("seed", "reference", "reference_sha1", "anchor_colour")

#: How root noise is actually suppressed on this path. Not a config field.
NOISE_SUPPRESSION = "search_with_root(state, add_noise=False)"


class E4ReferenceError(Exception):
    """An E4 task cannot produce the qualified reference construction."""


def reference_colour(task: Dict[str, Any]) -> str:
    """The colour OUR side plays: the opposite of the anchor's."""
    anchor = task.get("anchor_colour")
    if anchor not in ("red", "black"):
        raise E4ReferenceError(f"anchor_colour must be red or black, got {anchor!r}")
    return "black" if anchor == "red" else "red"


def rng_stream_seeds(task: Dict[str, Any]) -> Dict[str, int]:
    """The two generator seeds this task will actually use.

    Derived with the SAME masks the qualified path uses, imported from it. This
    is what binds a scheduled seed to the moves our side plays.
    """
    colour = reference_colour(task)
    seed = int(task["seed"])
    return {
        "colour": colour,
        "search_seed": seed ^ SeededReferenceAgent.SEARCH_MASK[colour],
        "readout_seed": seed ^ SeededReferenceAgent.READOUT_MASK[colour],
    }


def rng_witness(task: Dict[str, Any], draws: int = 4) -> Dict[str, Any]:
    """First `draws` values of each stream, for a TEST-ONLY seed.

    This is a REAL DRAW: it constructs both generators and takes values from them.
    It creates no experimental exposure all the same, and not because the draw is
    somehow lesser -- only because `TEST_ONLY_SEED_INTERVALS` can never enter a
    schedule, so a draw there strikes nothing off. On a schedulable seed the same
    call would strike that seed off permanently, with no model and no game. E4
    preflight attempt 3 learned that the expensive way, by taking witnesses over
    its own 32 scheduled seeds.

    THIS IS AN ALLOWLIST, NOT A DENYLIST, AND THAT IS THE POINT. The old version
    refused seeds it had been told about -- accounted, exposed, retired -- so any
    schedulable seed nobody had thought to list was drawn from and struck off in
    silence. That is exactly how 90000001 was witnessed into the frozen plan, used
    for the first real agent call, and still left absent from the registry. Now a
    witness may be taken only from the band reserved for exactly this, and an
    unrecognised seed is REFUSED rather than quietly consumed.
    """
    seed = int(task["seed"])
    if not seed_is_test_only(seed):
        raise E4ReferenceError(
            f"refusing to draw from seed {seed}: a schedule could use it, so drawing "
            f"would strike it off. Witnesses may be taken ONLY from "
            f"TEST_ONLY_SEED_INTERVALS ({list(TEST_ONLY_SEED_INTERVALS)}), which no "
            f"schedule may contain. Do not name a fresh 'synthetic' seed -- that is how "
            f"90000001 was drawn from twice and never recorded.")
    s = rng_stream_seeds(task)
    return {
        **s,
        "search_first": _first(s["search_seed"], draws),
        "readout_first": _first(s["readout_seed"], draws),
    }


def _first(seed: int, n: int) -> List[float]:
    r = random.Random(seed)
    return [r.random() for _ in range(n)]


def validate_task_structure(task: Dict[str, Any]) -> None:
    """Is this task WELL FORMED? Fields, pinned reference identity, colour.

    STRUCTURE ONLY. It asks nothing about seed availability, so it answers the
    same way forever: a schedule that was valid when it ran is still valid to
    parse, verify, classify and analyse after its seeds are gone. This is what
    lets a completed screen stay readable as historical evidence.
    """
    missing = [f for f in REQUIRED_TASK_FIELDS if f not in task]
    if missing:
        raise E4ReferenceError(f"task is missing {missing}; build_reference_agent would refuse it")
    ref = task["reference"]
    if ref not in REFERENCE_CHECKPOINTS:
        raise E4ReferenceError(f"unknown reference {ref!r}")
    pinned = REFERENCE_CHECKPOINTS[ref]["sha1"]
    if task["reference_sha1"] != pinned:
        raise E4ReferenceError(
            f"task reference_sha1 {task['reference_sha1']} != pinned {pinned}")
    reference_colour(task)          # raises on a bad anchor_colour


def validate_task_executable(task: Dict[str, Any]) -> None:
    """May this task be PLAYED NOW? Structure, then seed availability.

    A SEPARATE REQUIRED FUNCTION, deliberately not a `require_unspent=True`
    keyword on the structural one. A switch defaults, and a default that can be
    switched off is the failure mode this workstream keeps finding: every caller
    here must name which question it is asking, and the name is the answer.
    """
    validate_task_structure(task)
    seed = int(task["seed"])
    if seed in CONSUMED_SEEDS:
        raise E4ReferenceError(f"seed {seed} is recorded as already consumed")
    if seed_is_exposed(seed):
        raise E4ReferenceError(
            f"seed {seed} was EXPOSED -- it has been drawn from -- and cannot be scheduled")
    if seed_is_retired(seed):
        raise E4ReferenceError(
            f"seed {seed} belongs to a RETIRED block: its one-shot schedule completed, so "
            f"reusing any part of it would select tasks after seeing the result")


def _injective_streams(tasks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The seed->stream mapping must be injective, and structural.

    Two tasks sharing a generator stream would silently correlate two games that
    the schedule presents as independent. Nothing here draws: XOR only.
    """
    seeds = [int(t["seed"]) for t in tasks]
    if len(set(seeds)) != len(seeds):
        raise E4ReferenceError("duplicate task seeds")
    streams = [(rng_stream_seeds(t)["search_seed"], rng_stream_seeds(t)["readout_seed"])
               for t in tasks]
    if len(set(streams)) != len(streams):
        raise E4ReferenceError("two tasks derive the same generator streams")
    search_only = {s for s, _ in streams}
    readout_only = {r for _, r in streams}
    if search_only & readout_only:
        raise E4ReferenceError("a search stream collides with a readout stream")
    return {"n_tasks": len(tasks), "distinct_seeds": len(set(seeds)),
            "distinct_stream_pairs": len(set(streams)),
            "search_readout_disjoint": True}


def validate_schedule_structure(tasks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Is this schedule WELL FORMED? Independent of what has since been spent."""
    if not tasks:
        raise E4ReferenceError("empty schedule")
    for t in tasks:
        validate_task_structure(t)
    return _injective_streams(tasks)


def validate_schedule_executable(tasks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """May this schedule be RUN NOW? Structure, availability, and no test seeds.

    The test band is refused HERE rather than per task: building an agent on a
    test seed is fine and unit tests do it constantly, but a SCHEDULE is a set of
    games whose seeds must never have been drawn from -- and test seeds are drawn
    from by design. Scheduling is the operation that must refuse them.
    """
    if not tasks:
        raise E4ReferenceError("empty schedule")
    for t in tasks:
        validate_task_executable(t)
    for t in tasks:
        if seed_is_test_only(int(t["seed"])):
            raise E4ReferenceError(
                f"seed {t['seed']} is reserved for tests and is drawn from freely; it may "
                f"never appear in a schedule")
    return _injective_streams(tasks)


def build(task: Dict[str, Any], *, evaluator, config=None,
          capture: bool = False) -> SeededReferenceAgent:
    """The ONE construction path for the E4 screen. Delegates, never reimplements.

    AGENT CONSTRUCTION IS EXECUTION. It is the moment a seed becomes generators,
    so it asks the executable question, never the structural one.
    """
    validate_task_executable(task)
    return build_reference_agent(
        task=task, evaluator=evaluator, colour=reference_colour(task), config=config,
        capture=capture
    )


def frozen_settings() -> Dict[str, Any]:
    """The frozen research configuration, read from the qualified path itself."""
    cfg = eval_config()
    return {
        "eval_config": {f: getattr(cfg, f) for f in
                        ("board_size", "mcts_sims", "mcts_eval_batch_size",
                         "mcts_stall_flush_sims", "selection_mode",
                         "opening_temp_plies", "temp_high", "temp_low", "max_moves")},
        "noise_suppression": NOISE_SUPPRESSION,
        "noise_note": "cfg_from does NOT pass dirichlet_eps, so MCTSConfig keeps its "
                      "default 0.25; noise is suppressed at the call site instead. A "
                      "schedule that declares dirichlet_eps=0 is declaring nothing.",
        "search_mask": dict(SeededReferenceAgent.SEARCH_MASK),
        "readout_mask": dict(SeededReferenceAgent.READOUT_MASK),
        "readout_path": "eval_readout.select, never mcts.select_move",
        "agent_lifetime": "one instance per game; both streams advance across the game",
    }
