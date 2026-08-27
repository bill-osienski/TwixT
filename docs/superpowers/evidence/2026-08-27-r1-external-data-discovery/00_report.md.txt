# R1 — external data-source discovery: **NO_GO**

**Date:** 2026-08-27 · **Status:** READ-ONLY DISCOVERY REPORT. No corpus, model or bulk-data
download; no training, inference, games, seeds, JVM or T1j execution; no source, test or gate
changes. · Local, unpushed, docs-only.
**Result: `NO_GO`. Within scope, five searches and three successful metadata-page fetches found no
qualifying source.** `calib020_0001` remains incumbent, T1j remains evaluation-only, training stays
closed.

Basis: `main` @ `9cd9fef`. Evidence: `evidence/2026-08-27-r1-external-data-discovery/`.
Prior art: the E0 engine survey (2026-08-24) and R0 (`9cd9fef`).

> **The result is a bounded negative, not an impossibility proof.** A draft of this report argued
> that no corpus *could* carry π because SGF cannot represent it. **That was false**, and the
> correction is the most important thing in this document — see below.

---

## No surveyed source publishes π — but the format is not the obstacle

**SGF can carry a policy target.** FF[4] states that "Everybody is free to define additional, private
properties", which "may use one of the value types defined in this document or define their own value
type", and that unknown properties "should be preserved". A standard, general-purpose property `V`
already carries a real-valued node score across all games. A producer could publish visit
distributions in SGF, in a sidecar, or in any other format entirely.

What is true is narrower: **the TwixT profile defines no standard policy property, and no source
surveyed here publishes one.** Every archived TwixT game found is a **bare game history** — the
weakest of the three content grades R1 was asked to distinguish.

That still leaves candidate B's π problem unsolved *by these sources*. It does **not** establish that
π is unobtainable in principle, and a future engine author releasing self-play output would satisfy
it without any format change.

## Rules compatibility — we match no named ruleset

Ours, pinned in the ledger and **verified in code** (`_crosses_existing_bridge` iterates every bridge
with no owner filter, so a player's own links block too): 24×24, automatic links, **no link removal**,
**own-link crossing forbidden**, **swap off**.

| `RU` | defining feature | verdict |
|---|---|---|
| `STD` | link removal **and** manual link addition | we have neither |
| `PP` | links persist; own links may cross | crossing is what we forbid |
| `3M` | removal semantics undocumented | see the qualification below |

**A crossing PP link connects — an earlier draft said it did not.** The profile's own wording:
*"Links are not removed, but your own links may cross each other. This may result in a winning path
which loops across itself."* So the gap is **larger** than the draft claimed, not smaller:

- under `PP`, the bridge **forms and connects**, and paths may self-loop
- under ours, the bridge **never forms at all**

The same move sequence yields different bridge sets and different connectivity, so `PP` records
contain positions our engine cannot reach. The draft's conclusion survives; its explanation did not,
and it contradicted a source already fetched.

**On `3M`:** the profile says *"There are no recorded games, as far as David knows"* and *"David
welcomes any counterexample."* That is **author-bounded knowledge with an open invitation**, not a
specification finding. An earlier draft wrote "the specification states none exist", stripping the
qualifier.

**A documentation conflict the draft missed.** The current Little Golem docs say *"You may remove your
own links as part of a move to rearrange your network"* — contradicting the `PP` profile's "links are
not removed" — and offer three board sizes (24×24, 30×30, 48×48). So `RU[PP]` does **not** pin the
semantics of a recorded game. Provenance there is **weaker** than the draft credited, not stronger.

## The sources, at the width the evidence supports

| source | finding |
|---|---|
| **Little Golem** — largest live archive | rules mismatch (crossing, swap, board sizes); provenance weakened by the documentation conflict; contents **bare**; **no reuse permission found** and **no API or bulk export found** in the reviewed pages. **Could not be qualified within R1** — which is not the same as disqualified |
| **twixtbot** | **publishes no dataset** — `models/` and `src/` only; trained with crossings allowed |
| **Polygames** | E0: no TwixT implementation, 0 TwixT checkpoints among 1,133 |
| **3M corpora** | none known to the profile's author; counterexamples invited |
| **Kaggle / Zenodo / GitHub** | none surfaced by the listed queries |

**Independence was never the binding constraint.** Human Little Golem play *is* genuinely independent
of T1j and of our stack — it satisfies the one criterion R0 flagged as irreversible. What fails is
rules match, and, for the only live archive, licence and access that R1 **could not find** rather than
showed to be absent.

## Contamination and conversion concerns, recorded

1. **Rules conversion would be semantic, not syntactic.** A `PP` corpus encodes connections our engine
   cannot form. Using it means re-interpreting the game, not reformatting a file.
2. **T1j must never become a source.** Unchanged: training on the anchor converts it into a component
   of the thing it measures, with no second uncontaminated anchor. Nothing found here substitutes.
3. **Licensing is unresolved, not established either way.** No reuse grant was found; that is a reason
   R1 cannot qualify the source, not a finding that reuse is forbidden.
4. **A human corpus is expert moves at best, never π.** Even licensed and rules-matched, it supports
   supervised move prediction — a *different* objective from the AlphaZero policy target, proposed by
   no card and inheriting none of R0's pricing.

## Decision

**`NO_GO`.** No specific source qualified within scope. The negative is bounded: five searches and
three successful candidate metadata-page fetches, all listed with findings, plus one failed fetch
that was superseded by the canonical host. Supporting rules and format references inform the
interpretation; they are not additional candidate metadata pages.

## What would change this answer

1. **A source that publishes visit distributions** — an engine author releasing self-play output. No
   format change is needed for this; SGF private properties or a sidecar would carry it.
2. **A licensed, rules-matched archive** — or an explicit decision to accept `PP`-to-ours conversion
   *with* a written argument for why the reachability difference is tolerable. This report does not
   make that argument.
3. **Evidence that Little Golem offers an export route and reuse terms**, which R1 did not find but
   also did not rule out.
4. **A different objective.** Supervised move prediction on expert games would need its own card.

## Corrections carried in this report

| claim | status |
|---|---|
| "no corpus carries π **because the format cannot represent it**" | **withdrawn** — FF[4] permits private properties with custom value types and requires preserving unknown ones; standard `V` already carries a real node value. Replaced with: *no surveyed source publishes π* |
| a crossing `PP` link "is placed but simply does not connect" | **corrected** — the profile says such a path may win and may loop; the bridge connects. The rules gap is larger, not smaller |
| "the specification states no `3M` games exist" | **corrected** — *"as far as David knows"*, with counterexamples invited |
| Little Golem "no licence" / "no API or bulk export" | **narrowed** — neither was **found** in the reviewed pages; absence was not established |
| Little Golem provenance graded "GOOD" | **downgraded** — current docs contradict the `PP` profile on link removal, so `RU[PP]` does not pin semantics |
| `WEB SEARCHES (4)` heading over five entries | **corrected** — counts are now computed from the list, not typed |
