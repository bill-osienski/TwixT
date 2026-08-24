# E0 — external-anchor candidate survey

**Date:** 2026-08-24 · **Status:** E0 COMPLETE — read-only discovery, **with one disclosed scope exception**. One pilot candidate proposed.

> **SCOPE EXCEPTION, stated here rather than buried:** one source file (T1j `Board.java`) was
> **briefly written to the session scratchpad** so it could be grepped, then deleted. **Nothing was
> retained.** Saving source to disk is E1's job, not E0's, so this card cannot claim "online reading
> only" — and an earlier draft wrongly reconciled that by quietly dropping the word "downloaded"
> from its scope line, which papered over the exception instead of disclosing it. The exception does
> **not** invalidate the findings: nothing was retained, installed, executed, adapted or played, and
> the load-bearing source was subsequently **re-read through the authorized online path at its
> pinned revision** (integrity item (a) below). Full detail: integrity item (b).

· **Scope: online reading, plus the single disclosed exception above. Nothing retained, cloned,
installed, executed, adapted or played. No seeds consumed.**
· **E1–E4 are NOT authorized by this document.**

Basis: `main` @ `c7de3d4`, clean, local == remote. All sources read **2026-08-24**.
Predecessors: [`2026-08-22-twixtbot-anchor-pilot-card.md`](2026-08-22-twixtbot-anchor-pilot-card.md)
(anchor REJECTED, ledger do-not-repeat **#52**) and
[`2026-08-24-capacity-headroom-c0-card.md`](2026-08-24-capacity-headroom-c0-card.md)
(capacity fallback CLOSED, feasibility no-go).

---

## CORRECTIONS — review round 1, 2026-08-24

Four claims exceeded the evidence. All are corrected in place; the two integrity items below were
found while correcting them and are disclosed rather than quietly fixed.

| # | claim as first written | why it was wrong | replaced by |
|---|---|---|---|
| 1 | T1j has a "**genuine monotone strength dial**"; its dial "**demonstrably**" moves strength | E0 executed nothing. The source establishes a **controllable search-effort parameter**, not that raising it improves playing strength. OpenSpiel's non-saturation was likewise marked `✗` on a prediction, not a measurement | criterion 8 is now **controllable search effort** (verified) with **strength response unmeasured** (§A, §matrix); OpenSpiel criterion 9 is `~`, not `✗` |
| 2 | "twixtbot is the **ONLY** independently trained neural TwixT engine with obtainable weights"; "no weights **exist**"; "**none is available**" | a bounded web survey cannot establish non-existence. GitHub topic membership is **opt-in and not comprehensive**, so five tagged repositories is not an existence proof | every such claim is now bounded to **"identified in the searched sources as of 2026-08-24"** |
| 3 | "a handful of games at minimum and maximum depth **answers** it; 128 games does not answer it any better" | a small endpoint screen detects **blatant** saturation cheaply but **cannot conclusively establish non-saturation**, and 128 games genuinely do give greater statistical resolution | E4 is now described as a **preregistered staged screen with explicit early-stop and an explicit inconclusive outcome** (§recommendation) |
| 4 | sources cited by mutable repo/file name; licence given as "GPL-3.0" | no permalinks, revisions, release identity or access dates. The T1j **project page says only "GPL"** | every source is **pinned to a revision** with an access date (§sources); the licence claim is now sourced to the **LICENSE file at the pinned SHA**, which states GPL **version 3** |

### Integrity items found while correcting

**(a) My source reads went through a ref that does not exist.** I fetched
`raw.githubusercontent.com/johannesSchwagereit/T1j/**master**/…`, but T1j has only a `main` branch —
`git/ref/heads/master` returns **404**. I therefore re-read every load-bearing file at pinned
`main@b572ed2` through the contents API, unmediated by a summariser, and grepped for the exact
identifiers. **All claims held**, and the `Board.java` evidence turned out *stronger* than first
written (see §A). The first-round reads were nonetheless obtained through a URL that should not
have resolved, and that is recorded rather than glossed.

**(b) I briefly wrote one source file to the scratchpad to grep it, then deleted it.** Acquiring
source to disk is **E1's job, not E0's**. Nothing was retained, and the same content is obtainable
by reading; but it crossed the line the authorization drew, and is disclosed here.

---

## Result

**One candidate is worth piloting: T1j.** It is the only surveyed engine that is independent of our
lineage, licensed, source-available, exposes a controllable search-effort parameter, and —
decisively — **cannot suffer the failure that killed twixtbot**, because it has no neural network
and therefore no rules-used-during-training to mismatch.

**The headline finding is a bounded negative.** *No second publicly obtainable trained neural TwixT
engine was identified in the searched sources as of 2026-08-24.* Every other candidate identified
is a classical engine, a rules implementation with no player, closed source, or does not implement
TwixT. This is a statement about what a bounded survey found, **not a proof that none exists**:
GitHub topic membership is opt-in, the searched set is finite, and two sources could not be read at
all. If the programme's requirement is specifically a trained network of independent lineage, E0's
answer is that **this survey did not find a second one** — which is a reason to weigh whether to
continue, not a proof that continuing is futile.

---

## Candidate matrix

Ranked. `✓` supported by evidence read this session; `~` plausible or predicted but **unmeasured**;
`✗` fails on evidence; `n/a` structurally inapplicable — and for criterion 2 that is a **strength**.

| # | criterion | **T1j** | **OpenSpiel TwixT** | Ludii | Ai Ai | Polygames | twixtbot |
|---|---|---|---|---|---|---|---|
| 1 | Configurable rules compatibility | ✓ size/swap/side/start settable; **default board is 24** | ~ 24×24 + corners match; **no parameter disables swap** | ? TwixT presence unconfirmed | ? | ✗ no TwixT | ✓ (was) |
| 2 | Rules used during training | **n/a — no net** | **n/a — no agent ships** | n/a | ? | — | ✗ **trained with crossings allowed** |
| 3 | Independent provenance | ✓ Schwagereit, unrelated | ~ DeepMind repo, **TwixT code upstreamed from the twixtbot-ui author** | ✓ | ✓ | ✓ | — excluded |
| 4 | Licence | ✓ **GPLv3** (LICENSE at pinned SHA) | ✓ Apache-2.0 | ✗ **CC BY-NC-ND 4.0** | ✗ closed source | ✓ MIT | ✓ MIT |
| 5 | Obtainable model artifacts | ✓ n/a — engine *is* the artifact | ✗ **none published in the repo** | ? | ✗ | ✗ **0 TwixT among 1,133 checkpoints** | ✓ (was) |
| 6 | Reproducibility | ~ fixed-ply mode looks deterministic, **unmeasured** | ✓ seeded research library, has tests | ? | ✗ | ✓ | ✓ (was, measured 25/25) |
| 7 | Headless operation | ~ **engine file has 0 Swing/AWT references**, driving path untraced | ✓ no GUI exists at all | ~ | ? | ✓ | ✓ (via injected sink) |
| 8 | **Controllable search effort** | ✓ fixed ply or fixed time, iterative deepening — **strength response unmeasured** | ~ MCTS simulation budget — **strength response unmeasured** | ~ | ? | ~ | ✓ dial existed, but **measured not to move the score** |
| 9 | Plausible non-saturation | ~ **unmeasured; may be too weak** | ~ **unmeasured; predicted far too weak** | ? | ? | — | ✗ **measured 1.000 at every setting** |

Criterion 8 was renamed. "Genuine strength control" smuggled in a strength claim; what E0 can
establish is whether an engine **exposes a search-effort parameter**, not what strength that
parameter buys. Only twixtbot has a *measured* entry in rows 8 and 9, and it is a failure.

---

## The two serious candidates

### A. T1j — PROPOSED PILOT

Johannes Schwagereit, Java. Repository `github.com/johannesSchwagereit/T1j`, pinned at
`main@b572ed21f1f6b08491f69db32f531fc4b5f50fcd`; sole release tag `current`, published
2022-01-09. The project page states the program dates from 2006–2010 and is "No longer being
developed". **There is no `master` branch** — see integrity item (a).

**Licence.** The project page says only "T1j is licenced under the GPL", with no version. The
repository's `LICENSE` at the pinned SHA (blob `f288702d2fa1`, 35,149 bytes) is titled
**"GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007"**, and GitHub's metadata reports
`GPL-3.0`. The version claim rests on the LICENSE file, not the project page.

**Rules surface** — `MatchData.java` (blob `8ec82c977e1b`), lines 23–29, each also persisted via
`userPrefs` at lines 63–68:

```java
protected int     mdYsize    = Board.DEFAULTDIM;   protected int     mdXsize  = Board.DEFAULTDIM;
protected boolean mdYhuman   = true;               protected boolean mdXhuman = true;
protected boolean mdYstarts  = true;               protected boolean mdPieRule = true;
```

Board size, swap (`mdPieRule`, settable false to match our rules), which side the engine plays, and
the starting colour are all configurable. `Board.java` (blob `07d62cfa45ad`) gives
`MINDIM = 12`, `MAXDIM = 36` and **`DEFAULTDIM = 24`** — T1j's *default* board is already our size.

**Link crossing** — `Board.java`. `setBridge()` auto-creates links and then, at the comment on line
210, marks "all the 9 crossing bridges as illegal". Legality is decided by:

```java
public boolean bridgeAllowed(final int x, final int y, final int direction)
{  return field[x + MARGIN][y + MARGIN].bridge[direction] == 0; }
```

**The method takes no player or colour argument at all**, so it cannot distinguish own links from
the opponent's — crossing is forbidden regardless of owner. That is standard TwixT and matches our
own-link-crossing-forbidden rule, **the exact axis on which twixtbot's network was mismatched**.
(`removeBridge()` at line 281 decrements the same marks; it reads as search/undo machinery rather
than a player-facing rule, and that must be confirmed, not assumed.)

**Search-effort parameter** — `FindMove.java` (blob `a5ccc7869ff5`) and `GeneralSettings.java`
(blob `2a63154b1c16`). Lines 106–115 branch on `mdFixedPly` to take either `mdPly` (default 5) or
`mdTime`; line 121 runs `for (currentMaxPly = 3; currentMaxPly <= maxPly; currentMaxPly++)` around
an `alphaBeta(...)` call at line 126. So it is **iterative-deepening alpha-beta with a configurable
ply or time budget.**

> **What that does and does not establish.** It establishes a *controllable* search-effort
> parameter. It does **not** establish that raising it raises playing strength, and E0 ran nothing,
> so no strength response of any kind is measured here. Deeper alpha-beta search is *conventionally*
> stronger, but conventionally is not evidence — and search-depth pathology, evaluation-function
> artefacts and time-control effects are all real. **E4 must measure the response.**
>
> The contrast with twixtbot is one of *evidence*, not of demonstrated behaviour: for twixtbot we
> have a measurement that its `trials` dial did **not** move the score (1.000 at every setting,
> top-move visit share ~0.83 throughout); for T1j we have **no measurement either way**. What
> favours T1j is that its parameter is a different *kind* — search depth in a classical search,
> rather than visit count in an MCTS whose policy was already concentrated.

**Headless** — `FindMove.java` line 9 declares `package net.schwagereit.t1j`; lines 11–12 import
only `java.util.Map` and `java.util.HashMap`; a grep of the whole file for `javax.swing|java.awt`
returns **0 hits**. It exposes `getFindMove()`, `computeMove(final int player)` and
`setMatch(final Match matchIn)`. The GUI lives in separate `Gui*.java` / `StrengthDialog.java`.

> ⚠ **This is the same shape of claim I got wrong about twixtbot.** Zero GUI references in one file
> do not prove a driving path exists without a `Frame`; `Control`, `Match` and `GeneralSettings`
> are untraced, and `GeneralSettings` reads `userPrefs`. Criterion 7 stays `~`. **E2 must establish
> it by running, not by reading.**

**Strength, honestly.** The project page states "T1j will be beaten by most human players." That is
an informal author statement, **not a measurement, and it confers no Elo** — but it is one thing
twixtbot never gave us: a rough external human reference. Treat it as a prior for sizing the
screen, nothing more.

### B. OpenSpiel TwixT — viable infrastructure, not an anchor

`github.com/google-deepmind/open_spiel`, pinned at
`master@d7c4fc2dac825cb34b50131042a151b43c12edc5` (2026-08-12), Apache-2.0. TwixT is in the
official library (`open_spiel/games/twixt/`: `twixt.cc`, `twixt.h`, `twixtboard.cc`,
`twixtboard.h`, `twixtcell.h`, `twixt_test.cc`), described in `docs/games.md` as "Players place
pegs and links on a 24×24 square" and flagged 🔶 *implemented but lightly tested*. No GUI exists
anywhere, agents are seeded, and there is a test file — on licence, headless operation and
reproducibility it is the strongest candidate surveyed.

It fails as an anchor on two counts. **No weights are published in the repository** and none were
identified in this survey, so there is no independently trained player to anchor against. And
**training an agent inside OpenSpiel would be our training**, which reintroduces precisely the
sibling-vs-sibling circularity an external anchor exists to break.

On rules, from `twixt.h` at the pinned SHA: the only exposed accessors are `board_size()` and
`ansi_color_output()`, and `MaxGameLength()` is `board_size_ * board_size_ - 4 + 1` under the
comment `// square - 4 corners + swap move`. So a swap move is counted in the game length and
**no parameter is exposed to disable it** — a likely mismatch with our swap-off rule. Note the
countervailing detail: `NumDistinctActions()` is `board_size_ * board_size_` (576 at size 24), with
no separate swap action index, so how the swap is represented is **not settled by the header**.
Marked `~`, for E2/E3 to resolve.

Its geometry otherwise matches ours exactly: we exclude the 4 corners and each player's two border
lines (`scripts/GPU/game/board.py:19`), giving the 528 legal moves at ply 0 our eval replays
record. Its TwixT implementation was upstreamed from `stevens68/TwixT_for_open_spiel` — the author
of `twixtbot-ui`. That is rules code rather than a player, so it does not taint an anchor the way a
shared network would, but it is not fully independent of the twixtbot orbit.

---

## Eliminated, with the reason

| candidate | pinned at | reason |
|---|---|---|
| **Polygames** | `facebookarchive/Polygames` `main@eb5390e5` (2021-01-19), MIT, **archived** | **No TwixT implementation** in `src/games/`, and **0 TwixT checkpoints among 1,133** in the public list. The paper's mention of TwixT as a studied connection game is not reflected in the released framework. |
| **Ludii** | `Ludeme/Ludii` `master@8ba67089` (2025-02-21) | `LICENSE` is **"Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International"** — "You may not use the material for commercial purposes" and no distribution of modified material without permission. A poor basis for a research tool we would wrap. TwixT presence could not be confirmed: `details.php?keyword=Twixt` returns the portal homepage and the library listing truncated before `T`. |
| **Ai Ai** (Tavener) | not reachable | Freeware, **not open source**, so licence and reproducibility fail regardless. `mrraow.com` could not be read — expired TLS certificate. TwixT presence unconfirmed. |
| **twixtbot / twixtbot-ui** | — | **Excluded by ledger #52.** twixtbot-ui wraps the same network, so it is not a second candidate. |
| `stevens68/TwixT_for_open_spiel` | — | The upstream of candidate B. Game implementation, no player. |
| `MWPainter/Twixt-AI` | — | CS221 coursework, Python, 1 star, no licence stated, **no trained model** — `MCTreeSearch`, `PureMC`, `Minmax`, `HumanAgent`. No strength evidence. |
| `ioanTeulea/TwixtGame`, `probinso/twixt-ai-project`, `hendriku/rmaximus`, `danielehmig/TwixtProject` | — | Hobby, coursework or competition-client projects. No trained models, no strength evidence, no provenance. |

---

## Recommendation

**Pilot T1j.** It is the only candidate that clears independence, licence, availability and a
controllable search-effort parameter simultaneously, and the only one structurally immune to the
trained-under-rules mismatch that invalidated the last anchor.

**Its three open risks, in the order E1–E4 should attack them:**

1. **Non-saturation (criterion 9) decides everything, and it is unmeasured.** T1j may be too weak
   at every depth, as twixtbot was too strong at every `trials`. **E4 should be a preregistered
   staged screen**, not a single verdict: a small endpoint stage at minimum and maximum effort can
   **cheaply detect blatant saturation and early-stop**, which is all that is needed to kill a
   hopeless candidate — but it **cannot conclusively establish non-saturation**, and a larger match
   genuinely does give greater statistical resolution. The screen therefore needs **three**
   preregistered outcomes: *blatantly saturated → stop*, *clearly in band → proceed to a sized
   match*, and **explicitly inconclusive → decide on cost, not on a coin-flip reading.** Stage
   sizes, the band, and the early-stop rule must all be fixed before the first game.
2. **Headless (criterion 7) is unverified, and is where I was wrong last time.** `FindMove` has zero
   GUI references, but the path through `Control` / `Match` / `GeneralSettings` is untraced.
   Establish by running, never by reading imports.
3. **Determinism (criterion 6) is unverified.** Fixed-ply mode looks deterministic in shape, but
   `Zobrist.java` plus a transposition table can make results depend on search order and prior
   state. Measure it, as G2 did, rather than assuming it.

**A candid caveat on what even a successful pilot buys.** T1j is uncalibrated, exactly as twixtbot
was. Beating or losing to it yields an **ordering, not an absolute placement**, and no Elo. The
author's "beaten by most human players" remark is the only external grounding identified, and it is
informal. If the programme needs a genuinely calibrated anchor, **this survey did not identify
one** — and that should be weighed deliberately now rather than discovered again at the end of a
second pilot.

---

## Coverage — what was searched, so the gaps are visible

Searched 2026-08-24: general TwixT engine/bot queries; Polygames and its public checkpoint list;
the T1j project page, repository, release list and four source files; Ludii repository, `LICENSE`,
portal library and game-detail URLs; Ai Ai site and its BoardGameGeek game list; OpenSpiel
`docs/games.md`, the `games/twixt/` directory and `twixt.h`; the GitHub `twixt` topic; and targeted
searches for trained TwixT weights outside twixtbot.

**Why this cannot be a non-existence proof:** GitHub topic membership is **opt-in**, so the five
tagged repositories are a lower bound on what exists there; web search coverage is partial and
English-biased; engines hosted outside GitHub, on game servers, behind logins, or unpublished are
invisible to it; and two sources could not be read at all — `mrraow.com` (expired TLS certificate)
and `boardgamegeek.com` (HTTP 403).

**Not established, and out of E0's scope:** whether Ludii or Ai Ai implement TwixT; whether any
Little Golem or commercial TwixT bot is drivable; T1j's behaviour on a modern JRE; and **any actual
strength measurement of any candidate.**

### Sources, pinned

| source | revision / identity | read |
|---|---|---|
| `github.com/johannesSchwagereit/T1j` | `main@b572ed21f1f6b08491f69db32f531fc4b5f50fcd`; release tag `current`, published 2022-01-09; no `master` ref (404) | 2026-08-24 |
| — `LICENSE` | blob `f288702d2fa1`, 35,149 B — "GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007" | 2026-08-24 |
| — `src/net/schwagereit/t1j/FindMove.java` | blob `a5ccc7869ff5`, 10,141 B | 2026-08-24 |
| — `src/net/schwagereit/t1j/GeneralSettings.java` | blob `2a63154b1c16`, 1,875 B | 2026-08-24 |
| — `src/net/schwagereit/t1j/MatchData.java` | blob `8ec82c977e1b`, 4,605 B | 2026-08-24 |
| — `src/net/schwagereit/t1j/Board.java` | blob `07d62cfa45ad`, 18,724 B | 2026-08-24 |
| `johannes-schwagereit.de/twixt/t1j` | project page, undated content; states "GPL" without a version | 2026-08-24 |
| `github.com/google-deepmind/open_spiel` | `master@d7c4fc2dac825cb34b50131042a151b43c12edc5` (2026-08-12), Apache-2.0 | 2026-08-24 |
| `github.com/facebookarchive/Polygames` | `main@eb5390e57cc38e5287bf6dcfb420308a5995d194` (2021-01-19), MIT, archived 2022-03-02 | 2026-08-24 |
| `dl.fbaipublicfiles.com/polygames/checkpoints/list.txt` | 1,133 entries, no version identifier published | 2026-08-24 |
| `github.com/Ludeme/Ludii` | `master@8ba67089dd65129c9256031f00c5a32b4618da76` (2025-02-21); `LICENSE` = CC BY-NC-ND 4.0 | 2026-08-24 |
| `github.com/topics/twixt` | 5 tagged repositories; **membership is opt-in, not a census** | 2026-08-24 |
| `mrraow.com` (Ai Ai) | **unreadable** — expired TLS certificate | 2026-08-24 |
| `boardgamegeek.com` (Ai Ai game list) | **unreadable** — HTTP 403 | 2026-08-24 |
