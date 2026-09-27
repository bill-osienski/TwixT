# H5 — REPLICATION OF H4 — DESIGN CARD

> ⚠ **Read with §12 (corrections, 2026-09-27)** — it narrows "independent set"
> and "NOT controlled or observed", and withdraws one figure in §5. The text
> above §12 is otherwise as first committed (`aac5ce2`).

**Written 2026-09-25. DESIGN ONLY.** No code, no test, no gate, no seed survey,
no seed registration, no manifest, no JVM, no game, no push. The authorization
for this card covers the card and one local commit, nothing else.

> **The question H5 asks.** Under **exactly H4's frozen agents, rules and
> execution/randomization protocol**, in a **new, independent set of 296
> colour-reversed pairs**, does the incumbent's expected colour-balanced pair
> score again lie above 0.5?

H5 is a **directional replication** of H4. It is a study of its own: its own
seeds, identifiers, gates, manifests, evidence and analysis. It does **not**
re-estimate H4, amend H4, or combine with H4.

**What H4 established** (`docs/superpowers/evidence/t1j-h4-confirmatory/estimate.json`,
`3b607aa`): observed incumbent mean pair score **0.5929** over **296** pairs;
preregistered raw nominal two-sided 95% Hoeffding interval **[0.5140, 0.6718]**,
which, **under its declared independence assumption**, supports an expected
score above 0.5 for the frozen empty-board protocol. It does not establish a
universal ranking, does not show that T1j's unseeded JVM draws are independent,
and does not explain the difference from H3 (0.7348, random six-ply positions).
**That result stands whatever H5 finds** (§6).

---

## 1. What is carried over UNCHANGED from H4

Everything that defines the game and the protocol is H4's, by reference, and is
**not re-decided here**. Values below were read from source on 2026-09-25.

| item | H5 value | authority |
|---|---|---|
| estimand | expected colour-balanced pair score for the incumbent, over independent pair bundles from the frozen protocol, games from the empty board | H4 replacement card §1.1 |
| pair | Arm A incumbent Red (moves first), Arm B T1j Red; the two games share only a bundle index; **no CRN coupling claimed** | replacement §1.2 |
| points | win 1, loss 0, **cap 0.5**; pair score ∈ {0, 0.25, 0.5, 0.75, 1} | replacement §1.2; analysis card §3.2 winner encoding |
| incumbent | frozen argmax agent, `SELECTION_MODE = "argmax"`, `MCTS_SIMS = 400` | `h2_match_rules.py:52`, `:57` |
| T1j | depth `T1J_MDPLY = 6` with `mdFixedPly=True`, **including its native early moves** (the §4B acceptance mode, `h4_acceptance=true`) | `l0_match_rules.py:106`; §4B card |
| cap | `PLY_CAP = 280` total plies | `l0_match_rules.py:109`; replacement §1.6 |
| process lifecycle | fresh JVM per T1j move, fresh JVM per binder call, no reuse, classes compiled once per run | replacement §3.1–3.2 |
| randomization sources | incumbent streams per arm from the seed; **T1j native entropy NOT controlled or observed** | replacement §3 |
| persistence | every move, every process record, re-derivable coherence digests, transcript digest recomputed | replacement §7; runner card §3–4 |
| multiplicity | identical trajectories are **data**, retained at full multiplicity; no duplicate rule, no uniqueness floor | replacement §1.3–1.4; analysis card §0.1 |
| operational failures | the H4 runner card §6 list, unchanged; a native-initial reply and a cap are **not** failures | runner card §6 |
| per-segment deadline | **14,400 s** (`SEGMENT_DEADLINE_S`) | `h4_runner.py:142`; replacement §5.4 |

**Toolchain and incumbent identity are bound, not described.** H5's production
qualification (§8) must show that H5's bound `t1j_runtime` content and
`incumbent_identity` **equal the values in H4's committed study manifest**
(`docs/superpowers/evidence/t1j-h4-manifests/study_manifest.json`) in every
field except the design label and H5's own identifiers. Any other difference is
a **STOP that comes back to the user as an amendment**: a replication run on a
different runtime or incumbent is not this study.

🔴 **No change to agent, adapter or play behaviour is part of H5.** H5 adds
identifiers, gates, schedules, manifests and its own analysis; if implementation
finds that play behaviour must change, that is an amendment to this card, not a
quiet edit.

---

## 2. Size, segments and seeds — FROZEN

| | |
|---|---|
| size | **296 pairs / 592 games** |
| segments | **four**, each **74 complete pairs / 148 games**; arms adjacent; no pair crosses a segment boundary |
| seeds | **four separate blocks of 148**, one per segment, **one seed per game** |
| seed → game | within a block, pair *k* (0-based, schedule order): Arm A `lo + 2k`, Arm B `lo + 2k + 1` — H4's rule (step-4 card §2) |
| seed reach | the incumbent only; T1j is unseeded |

**Why 296 and not a larger n.** H5 keeps H4's size and four-segment shape
deliberately: it makes H5 a closer replication, and 148 games is an operational
unit that has **actually completed four times under exactly this protocol** (§7).
The size is **not** chosen to make a positive result likely (§5) and **will not
be changed** — not before a segment, not between segments, not after.

---

## 3. Primary analysis — FROZEN

* **pair score** as in §1; **every valid pair counted with its multiplicity**;
* **primary**: mean pair score over all 296 valid pairs, caps scoring 0.5;
* **interval**: two-sided **nominal** 95% Hoeffding,
  `mean ± h`, `h = sqrt(ln(2/0.05) / (2 · 296))` = 0.0789380472 (to 10 d.p.);
* intervals are **raw** — never clipped to [0, 1];
* pair formation, validation and the winner encoding are H4 analysis card §3.2,
  unchanged; any malformed pair is an integrity fault and the analysis
  **refuses** — it never drops a pair;
* the analysis runs only on **exactly the four H5 segment files**, each
  complete, committed, unmodified and bound to its own H5 manifest entry; an
  incomplete H5 is **not** a smaller H5 and yields **no estimate**.

### 3.1 🔴 The decision rule — on the UNROUNDED, COMPUTED bounds

| computed raw bounds | H5 verdict |
|---|---|
| **lower bound strictly > 0.5** | **DIRECTIONAL REPLICATION** |
| **upper bound strictly < 0.5** | **OPPOSITE DIRECTION** |
| anything else (including a bound exactly 0.5) | **INCONCLUSIVE** |

The rule is stated on the bounds **as computed** from the mean and `h`, never on
a rounded threshold such as "mean > 0.578938". Descriptively only: pair-score
means lie on the lattice *j*/1184, so the smallest mean that replicates is
686/1184 ≈ 0.57939 and the largest that is opposite is 498/1184 ≈ 0.42061; no
attainable mean lies within 3.9×10⁻⁴ of either boundary, so floating-point
rounding cannot flip a verdict. The code uses the computed bounds regardless.

🔴 **No threshold, size or rule may be changed** after any H5 game exists. An
INCONCLUSIVE H5 is a **result**, reported as one — not a failed run and not a
reason to extend.

---

## 4. Descriptive sensitivities — reported, never confirmatory

None of these may change the verdict of §3.1, and each is **labelled
descriptive** in the report:

1. **Cap-free**: excludes **whole pairs** containing any cap; its own mean, `n′`
   and raw interval; `n′ = 0` → "not estimable", no mean, no interval (analysis
   card §3.3, unchanged). It is **conditional** on the observed caps.
2. **Per segment, in run order**: for segments 0, 1, 2, 3 in the order they ran —
   mean pair score (n = 74 each), cap-affected pairs, cap games, `segment_end`
   wall time. **No per-segment interval, no test, no trend statistic**; segment
   order is the time order.
3. **Concentration**, as H4 confirmatory reported it: unique complete games by
   arm, maximum game frequency, first-*k*-ply prefix frequencies for
   *k* = 1 … 6, identical pair-level trajectory tuples, answering-routine
   frequency by ply and colour, unique outcome patterns.
4. **H4 beside H5**: H4's and H5's means, raw intervals, cap counts and
   concentration summaries **side by side**, with the difference of the two
   means stated as a number. **No interval or test on the difference, and no
   pooled estimate is computed** by any H5 module. A pooled analysis, if ever
   wanted, is a separate later card; it is never H5's primary.

**No other sensitivity is frozen.** Any further one needs an amendment to this
card **before** any H5 outcome is read.

---

## 5. Operating characteristic — a WARNING, not a sizing input

Recorded before any H5 seed exists, so it cannot be mistaken for hindsight.
From H4's own 296 pair scores (mean 0.59291, population SD 0.33739; read from
the four committed H4 segment results after H4's analysis was published):

| assumed true mean | P(H5 lower bound > 0.5), normal approximation |
|---|---|
| 0.5929 (H4's observed mean) | ≈ **0.76** ~~(a bootstrap of H4's pair scores, 20,000 resamples, gives 0.756)~~ — withdrawn, §12 C3 |
| 0.55 (inside H4's interval) | ≈ **0.07** |

The first row **assumes H4's estimate is the truth**, which is likely
optimistic: H4 cleared its bar narrowly (lower bound 0.514), and an estimate
that just cleared a bar tends to overstate the true value. So **an INCONCLUSIVE
H5 is a live, informative outcome even if H4 is right**. This is why §3.1 names
it as a verdict. The calculation is **not** used to size H5 (§2).

---

## 6. Interpretation — frozen before any number exists

| H5 verdict | what may be said |
|---|---|
| **DIRECTIONAL REPLICATION** | in a second, independent set of 296 pairs, the incumbent's expected score is again above 0.5 in empty-board direct play **under this joint execution protocol**, under the declared independence assumption. It says nothing about the **size** of the difference beyond H5's own interval, and it does not make H4's thin margin wider |
| **OPPOSITE DIRECTION** | H5's interval lies below 0.5 under this protocol, **conflicting** with H4. Both results are reported as they stand; neither is discarded, and the conflict is **not** resolved by pooling. H5 alone cannot localize a cause |
| **INCONCLUSIVE** | H5 does not replicate H4's direction at its preregistered precision. It is **not** evidence of parity and **not** a refutation of H4 (§5) |

**In every case:**

* 🔴 **H4's published result stands unchanged.** H5 never edits, re-analyses or
  re-labels H4.
* 🔴 **Replication does NOT verify independence.** Both intervals remain
  **conditional on the declared assumption** that seed bundles and T1j's
  unseeded fresh-JVM draws are independent. H5 repeating H4's direction adds a
  second draw under the same assumption; it is not a test of it. This caveat is
  stated beside every H5 interval, whatever the verdict.
* **No explanation of H3.** H5, like H4, is a different estimand from H3's
  random six-ply positions.
* **No pooling** with H1, H2, H3, L0 or H4; **no claim** about human play, other
  configurations, or either engine outside this protocol.
* H4 replacement card §9's Arm B caveat carries over: in Arm B T1j moves first
  at exactly the plies where it does not search, and the design cannot separate
  that from general strength.
* H5 and H4 ran at **different times** on the same machine. Only the toolchain
  and incumbent are bound (§1); other machine state is part of the protocol and
  is not controlled.

---

## 7. No pilot — and why none is needed

**H5 has no pilot and plays no game before its four segments.** There is no
outcome-bearing pilot, no "sanity" game and no early strength game of any kind.

The H4 pilot existed to establish **timing and cap feasibility** for a protocol
that had never run. For the identical protocol that evidence now exists from
**four completed full segments**, the same size H5 will run:

| H4 segment | record | commit | `segment_end.total_s` | cap-affected pairs / 74 |
|---|---|---|---:|---:|
| 0 | `docs/superpowers/evidence/t1j-h4-study-segment0/results.jsonl` | `09c2e00` | 8,023.4 | 4 |
| 1 | `…/t1j-h4-study-segment1/results.jsonl` | `17f49ba` | 8,461.6 | 5 |
| 2 | `…/t1j-h4-study-segment2/results.jsonl` | `9f21a2d` | 9,195.8 | 10 |
| 3 | `…/t1j-h4-study-segment3/results.jsonl` | `a68f234` | 8,301.4 | 6 |

* **Timing:** 8,023–9,196 s per 148-game segment, total 33,982 s ≈ 9.4 h; the
  slowest used 64% of the 14,400 s deadline and was below H4's 10,800 s
  projection ceiling. This is **planning evidence, not a guarantee** for H5.
* **Caps:** 25 cap-affected pairs of 296 (26 cap games), per the H4 confirmatory
  estimate; the per-segment counts above sum to it.
* The H4 pilot (`t1j-h4-pilot/`, decision `PROCEED` with no rule fired,
  `2b1cb56`) is **not** reused as H5 data and not needed as H5 evidence.

**What replaces the pilot's rules in H5.** Only the operational-failure VOID and
the 14,400 s deadline (§1) can stop a segment. **Caps and concentration are
results, never stop rules** (replacement §5.4, full-study clause). Because there
is no pilot decision, H5's confirmatory analysis has no proceed/stop artifact to
check; that precondition is replaced by the gate and the four durable, bound
segment files (§3). **A pilot artifact of any stage — H4's included — is refused
as H5 input.**

**A no-game production qualification is still required** before any H5 seed is
used (§8, §10): it plays no game, makes no T1j move or replay, uses no research
seed and produces no outcome.

---

## 8. The H5 execution path — requirements for the implementation card

The implementation is designed in a **separate card** and built only under its
own authorization. It must satisfy:

1. **Reuse the tested behaviour.** The play path, adapter acceptance mode,
   process lifecycle and persistence are H4's, exercised in 624 H4 games
   (32 pilot + 592 study) with no adapter refusal. §4A/§4B are not re-run
   unless play-path code changes, which §1 forbids without an amendment.
2. **H5's own identifiers**, fixed now because manifests will name them:
   * design label `H5_STUDY`, segment index 0–3;
   * pair ids `h5s{segment}-00 … h5s{segment}-73`, task ids `…-A` / `…-B`;
   * evidence directories `docs/superpowers/evidence/t1j-h5-study-segment<k>/`,
     manifests `docs/superpowers/evidence/t1j-h5-manifests/`, estimate
     `docs/superpowers/evidence/t1j-h5-confirmatory/`.
   Module, gate and test names are fixed in the implementation card.
3. **New H5 gates, created CLOSED**, at least: the production qualification,
   segment execution, and confirmatory aggregation. **No H4 gate can open an H5
   action and no H5 gate an H4 one.** The gate inventory count is derived at the
   time the gates are added, not stated here.
4. **The retired H4 material cannot run again, and H5 cannot run H4's.** H5
   refuses H4 seeds (all RETIRED), H4 design labels, H4 schedules, H4 manifests
   and H4 evidence directories; H4 entry points refuse H5's. Each refusal is
   **proven with a negative control** through the real entry point.
5. **Gates bind:** checked first, before any file is opened (input-read spy, as
   the H4 analysis card §4); **no default that switches a check off**; the
   production entry points **accept no seam**, not even paths; every CLI is
   qualified as a **fresh subprocess** by exit code.
6. **Synthetic fixtures prove**, before any real run: gate refusal, replay
   integrity, multiplicity (*k* identical pairs count *k* times), outcome
   blinding, one-shot create-only destinations (an occupied directory refuses,
   including one left by a VOID), manifest binding field by field, and the §3.1
   rule at its boundaries (a bound exactly 0.5 is INCONCLUSIVE). Fixtures are
   unmistakably not evidence (negative seeds, fixture headers).
7. **No H3 code** in any H5 analysis, as H4 analysis card §0.1.
8. **Every new check is shown to reject** in the injected-defect harness, with
   a clean baseline.

---

## 9. Seeds — a FRESH survey and proof

* **Fresh survey, fresh proof.** H5's candidate blocks come from a **new
  read-only survey** of the registries as they stand after H4's closing edit
  (`0aec706`). H4's candidate ranges, survey output and pinned exemption list
  are **not inherited**, and H4's survey result is not evidence about any H5
  candidate.
* **Method:** the collision proof as H4 left it (v15 + v16 provenance, step-4
  card §5 and amendments 6–7) — all registries, `CONSUMED_SEEDS`, named spent
  blocks, H3 generation ranges, XOR-mask derivations, gap floor = own size,
  literals judged per occurrence, differential negative controls — applied as a
  **new H5 proof with H5's own pinned records**. If that method fails against
  this card, **work stops and the question comes back** rather than the method
  being adjusted and re-run.
* **No H4 seed is reused** — all 624 are RETIRED and the runner refuses them —
  and **no H4 game is reused**.
* **Four separate blocks**, so one segment's VOID cannot block the others'
  accounting (H3's lesson, step-4 card §2).
* **Register, then re-qualify.** The shared registry `e4_screen_reference.py` is
  hashed into the bound code; the registry edit therefore comes **before** the
  final production qualification, and **no bound file changes after it**.
* **Use-recording is deferred** until H5 closes (step-4 card §2.1, unchanged):
  until then the occupied, fsynced, committed evidence directories are the
  one-shot record; one closing edit then marks exposure and retirement.

---

## 10. Execution

* **Four segments, each separately authorized, each one-shot.** Open that
  segment's gate in its own commit, run once, restore the gate on **every** exit,
  commit the segment's evidence with the restoration **before** the next segment
  is authorized.
* **A VOID ends the H5 line.** It is never retried with the same seeds, the
  study is never shrunk, and no replacement block is drawn under this card. A
  VOID H5 has **no confirmatory estimate**; unstarted blocks are RETIRED at the
  closing edit; any new attempt is a new study with a new card and fresh seeds,
  never pooled with the VOID H5's completed segments.
* **Outcome blinding until aggregation:** between segments only record types,
  the header, `segment_end`, process pids and the blinded trace are read; no
  `game_result` is opened and `load_games` is not run until the aggregation step
  is authorized.
* **Aggregation only after all four segments are durable**, under its own gate
  and authorization; then the closing seed edit; then publication beside H4.

### 10.1 The staged order — one authorization each

| step | what |
|---|---|
| **1** | **this card** — design only |
| 2 | implementation card: modules, gates, manifests, analysis entry point, controls — design only |
| 3 | implement, gates CLOSED, fixtures only, injected-defect controls |
| 4 | no-game production qualification of the built path, compared with H4's bound runtime and incumbent (§1) |
| 5 | fresh read-only seed survey |
| 6 | collision proof + registration of the four blocks + runner registry checks |
| 7 | re-qualification of the **final** code state (no game) — nothing bound changes after it |
| 8 | manifests, re-derived and compared, committed |
| 9–12 | segments 0, 1, 2, 3 — one each |
| 13 | confirmatory aggregation |
| 14 | closing seed edit |
| 15 | optional replay export |
| — | push — only on explicit instruction |

🔴 **This card becomes a bound card** once manifests exist: its hash enters the
H5 manifests alongside the implementation card(s) and the H4 cards it
incorporates by reference. Any amendment after step 7 voids that qualification
for binding.

---

## 11. What this card does not do

It writes no code or test, creates no gate, surveys or registers no seed, writes
no manifest, starts no JVM, plays no game, reads no H5 outcome (none exists), and
pushes nothing. It does not edit H4's cards, code or evidence, and it does not
design an estimand that avoids the independence assumption — that would be a
different study, with a different randomization protocol, under its own card.

---

## 12. Corrections — 2026-09-27, before the card is frozen

Reviewed by the user 2026-09-27. The text above is left as first committed
(`aac5ce2`) except the one strike-through in §5; these corrections govern where
they differ.

**C1 — "independent set" (the opening question; §6, DIRECTIONAL REPLICATION
row).** It means **fresh, non-overlapping pairs**: new seeds, new games, no H4
seed, game or record reused. It does **not** mean the pairs' outcomes are
observed to be independent. Independence of the outcomes — across H5's pairs,
and between H5 and H4 — remains the **declared assumption** of §6, not an
observed property.

**C2 — "T1j native entropy NOT controlled or observed" (§1 table).** What is
unobserved is T1j's **RNG seeds** (each fresh JVM's `new Random()` state). T1j's
**realized moves are persisted**, every one, with their process records (§1,
persistence row); the persisted moves, not a seed, are the reproducibility
record of every H5 game.

**C3 — the §5 bootstrap figure is withdrawn.** "0.756" was computed without
recording enough to reproduce it (resample order, generator and seed), so it is
struck. The §5 warning stands on the normal approximation alone: ≈ 0.76 at a
true mean of 0.5929 and ≈ 0.07 at 0.55, from H4's mean 0.59291 and population SD
0.33739 of its 296 pair scores, with `h` as in §3.

**Also recorded at this review:** the additions made in `aac5ce2` beyond the
four frozen points — the runtime/incumbent equality with H4's manifest (§1), the
fixed identifiers (§8), the descriptive H4–H5 difference with no test or pooling
(§4), refusal of any pilot artifact as input (§7), and the staged order (§10.1)
— are approved. The two no-game qualifications in §10.1 have **distinct jobs**:
step 4 (**early**) finds production-path faults before any H5 seed is
registered, and binds nothing; step 7 (**binding**) qualifies the final code
state after registration and is the only record the manifests may bind to. The
implementation card (`2026-09-27-t1j-h5-implementation-card.md`) makes that
distinction mechanical.
