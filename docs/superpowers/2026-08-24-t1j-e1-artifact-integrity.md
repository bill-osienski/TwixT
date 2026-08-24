# E1 — T1j artifact acquisition and static integrity

**Date:** 2026-08-24 · **Status:** E1 COMPLETE. **Outcome 1 — OFFICIAL-RELEASE-QUALIFIED for E2
execution; executable-source correspondence UNVERIFIED.**
· **Scope: acquisition into isolated scratch, hashing, archive listing, textual metadata, and static
source reading. No installation, compilation, class-file extraction, `javap`, class loading, Java
execution, adapter work, games or seeds.**
· **E2–E4 are NOT authorized by this document.**

Basis: `main` @ `016be24`, clean, local == remote. Acquired and verified **2026-08-24**.
Predecessor: [`2026-08-24-external-anchor-e0-candidate-survey.md`](2026-08-24-external-anchor-e0-candidate-survey.md).
Hashes: `evidence/2026-08-24-t1j-e1/hashes.txt`.

**Nothing acquired was placed in this repository.** The clone and the JAR live only in session
scratch, outside the repo; this card and the hash file are the only committed products.

---

## Outcome

**1 — Official-release-qualified for E2 execution. Executable-source correspondence unverified.**

> **Corrected in review round 1.** This section first said the JAR "binds to the pinned source at
> three independent levels". That conflated two different things. What the evidence establishes is
> that **the official release artifact is adequately identified** and may proceed to E2. It does
> **not** bind the JAR's **executable bytecode** to the pinned Java source.

| level | evidence | what it establishes |
|---|---|---|
| **artifact identity** | sha256 `53ec95e4…`, 83,990 bytes, sole asset of the sole release of the official repository, acquired from `github.com/johannesSchwagereit/T1j` | **which bytes** we would run |
| **shipped resources** | all **6 of 6** non-class resources sha256-identical between the pinned source tree and the JAR, hashed in place | the JAR ships the same **resources** as the source tree |
| **class inventory** | all **21** source top-level classes present as `.class` entries; **0 orphan** classes; normalized diff empty | the JAR contains the same **set of class names**, no more, no less |
| **build manifest** | the JAR's `META-INF/MANIFEST.MF` and the committed `src/Manifest.mf` are **byte-identical** — both 192 bytes, sha256 `0bd77a8e…`, CRLF throughout | the source tree **contains a copy of** that manifest |
| **executable bytecode** | **not examined** — class-file extraction and `javap` are excluded by this authorization | **nothing** |

**Three limits on the above, stated so they are not read as more than they are.**

1. **Matching resources and class names is not a code binding.** Filenames and shipped assets can
   coincide while the compiled code differs arbitrarily. Nothing here constrains what
   `Board.class` actually does.
2. **An identical manifest does not prove which build produced the JAR.** `src/Manifest.mf` is an
   ordinary text file sitting in the source tree; that both copies are byte-identical shows the
   tree carries a copy of a manifest, not that *this* JAR was produced from *this* tree. It is
   consistent with that story, and with others.
3. **Consequently, every rules and control-flow finding in this card is a property of the
   SOURCE only.** The anchor would play the JAR's bytecode. The two are not shown to be the same
   thing.

**What E3 can and cannot do about it.** E3's per-ply state equivalence would **behaviourally
qualify the released artifact over the corpus it exercises** — every state actually reached is
checked against our engine. It **cannot** authenticate the binary: it says nothing about bytecode
outside the states played, and an untested state may still diverge. Behavioural qualification over
a tested corpus is a genuine and sufficient basis for using the artifact as an anchor; it is not
a proof of source identity, and must not be reported as one.

**On seeking a bytecode-level binding.** Rebuilding from the pinned source and comparing is **one
possible investigation**, not a decisive one: `javac` output varies by compiler version and
settings, so a **non-matching modern build would not disqualify the original binary**, and a
matching one is not obtainable — `build.xml` compiles with `source="1.4" target="1.4"`, which no
current JDK accepts, and its `init` target regenerates the manifest from `${user.name}` and
`${TODAY}`. A rebuild answers a different question than the one it appears to answer.

**Why the official JAR is nonetheless preferable to a modern rebuild for E2:** it is the artifact
the author published and the one that has been downloaded 73 times, whereas any build we perform
must change the compiler settings and so produces an artifact **nobody has ever run**.

---

## Identity and integrity

| item | value |
|---|---|
| repository | `github.com/johannesSchwagereit/T1j` |
| refs | `main` = tag `current` = **`b572ed21f1f6b08491f69db32f531fc4b5f50fcd`** |
| tag type | **lightweight** — a bare pointer, so it carries no tagger, date or signature of its own |
| `master` | **does not exist** (`git/ref/heads/master` → 404), confirming E0's integrity item (a) from a second, independent direction |
| tree | `6f1adcd6a3449a026c8c69b799248b6819017448` |
| parent | `61c81173b967bfa29b92c5eec69c8ad036ed607a`, "Initial commit", 2022-01-08T19:05:33+01:00 |
| commit | "Initial setup.", authored **and** committed 2022-01-09T09:51:04+01:00 |
| author | Johannes Schwagereit `<johannes.schwagereit@metro.digital>` |
| history | **2 commits total** — a 2022 upload of a 2006–2010 program, not a development history |
| clone | `git status --porcelain` **empty**; `git fsck --full` **clean** |

**On the author email.** The commit is signed off with a corporate address (`metro.digital`), while
the project page and README use `johannes-schwagereit.de`. The name matches and the release, tag
and repository are all under the same GitHub account, but **the commit's email does not
independently corroborate authorship**. Recorded as a fact, not treated as a discrepancy.

### Licence

`LICENSE`, 674 lines, sha256 `3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986`,
titled **"GNU GENERAL PUBLIC LICENSE / Version 3, 29 June 2007"**. This settles the version that
E0 could not: the project page says only "GPL".

**18 of 21** `.java` files carry an in-file GPL notice. The three without one are
`GeneralSettings.java`, `Stopwatch.java` and `Zobrist.java`. The repository-level `LICENSE` plus
the author's published statement cover the work; noted for completeness, not as a problem.

### Release artifact

| item | value |
|---|---|
| release | tag `current`, name **`v0.2`**, published **2022-01-09T08:59:12Z** |
| asset | `t1j.jar`, **83,990 bytes**, uploaded 2022-01-09T08:56:56Z, 73 downloads |
| sha256 | `53ec95e421db2531758142e9ee8ae49030f5345f5dc0c57b2ddb103fbd44e9b7` |
| sha1 | `064370a89b8361cdb400732d8942b2cc31d31437` |
| entries | 69, **every one dated 2007-03-30 21:45** |

Its manifest:

```
Manifest-Version: 1.0
Ant-Version: Apache Ant 1.6.5
Created-By: 1.5.0_09-b03 (Sun Microsystems Inc.)
Built-by: johannes
Built-On: March 30 2007
Main-Class: net.schwagereit.t1j.Control
```

**So the JAR is a 2007 build, uploaded in 2022.** It cannot have been produced *from* commit
`b572ed2`, which did not exist until 15 years later. E1 therefore asked the narrower question of
how the artifact relates to the *source tree* that commit records — and the outcome table answers
that at the level of **resources, class names and a manifest copy only**, not at the level of code.

The release was published **8 minutes after** the commit (`08:51:04Z` → `08:56:56Z` asset →
`08:59:12Z` publish), consistent with a single upload session. Note that GitHub sets a lightweight
tag's release `created_at` from the commit date, so that field is **not** independent evidence.

---

## Dependencies and build surface

**Zero external dependencies.** `build.xml` contains no `classpath`, no `lib`, no Ivy or Maven
reference — 0 matches for any of them. It compiles `src/**/*.java` with Ant's `javac` and jars the
result. Nothing is fetched at build time. For E2 isolation this is close to the best case.

Two facts from `build.xml` that matter later:

- **`source="1.4" target="1.4"`.** No current JDK accepts these. Any E2 build must raise them,
  which changes the artifact from what the author shipped.
- **`Main-Class: net.schwagereit.t1j.Control`**, and the `init` target *regenerates* the manifest
  on every build — so a rebuild cannot reproduce the committed manifest even in principle.

**Java requirement.** The JAR was created by `1.5.0_09` targeting 1.4, so its class files should be
major version 48. That is **inferred from `build.xml` and the manifest, not read from class bytes**
— reading class files is outside this authorization. **Whether a current JVM will load major-48
class files is an E2 empirical question and is deliberately not asserted here.**

---

## Rules surface, confirmed at the pinned revision

Everything E0 reported through the bad `/master/` URL was re-verified here against the clone at
`b572ed2`. All of it held.

- `Board.java` — `MINDIM = 12`, `MAXDIM = 36`, **`DEFAULTDIM = 24`**: the default board is already
  our size. `setPin()` auto-creates links via `setBridge()`, which marks "all the 9 crossing
  bridges as illegal".
- **`bridgeAllowed(final int x, final int y, final int direction)` returns
  `field[x + MARGIN][y + MARGIN].bridge[direction] == 0`** — it takes **no player or colour
  argument**, so it is structurally incapable of distinguishing own links from the opponent's.
  Crossing is forbidden regardless of owner: standard TwixT, matching our rule.
- `MatchData.java` — `mdXsize` / `mdYsize`, `mdPieRule` (default `true`, settable `false` for
  swap-off), `mdXhuman` / `mdYhuman`, `mdYstarts`; all persisted under the Preferences node
  `/net/schwagereit/t1j`.
- Source is **complete**: `Node`, referenced by `Board` and `Zobrist`, is an inner class in each —
  no missing file.

### The effort parameter's range

`GeneralSettings.correct()` clamps **only** `mdPly < 1 → 5` and `mdTime < 1 → 5`. **There is no
upper bound in code.** The GUI's own `StrengthDialog` offers `SpinnerNumberModel(5, 1, 10, 1)` for
ply and `(5, 1, 60, 1)` for seconds — so **1–10 ply is the range the author exposed to users**,
while a programmatic driver is not bounded by it. Useful for E4 sizing; still says nothing about
what strength any setting buys.

---

## The prospective headless path — feasible, but NOT the natural one

E0 marked this `~` on the strength of `FindMove.java` having no GUI imports. **That caution was
justified: the natural setup path does touch the GUI.**

**GUI-free (0 `javax.swing` / `java.awt` imports):** `Board`, `CheckPattern`, `Evaluation`,
`FindMove`, `GeneralSettings`, `InitialMoves`, `MatchData`, `Messages`, `Move`, `OrderedMoves`,
`Races`, `Stopwatch`, `Zobrist`.

**GUI-coupled:** `Control` (3), `GuiBoard` (2), `GuiMainWindow` (3), `LoadSave` (3), **`Match` (2)**,
`NewDialog` (12), `RightPanel` (4), `StrengthDialog` (6).

`Match` is the one that matters, because `FindMove.setMatch(Match)` requires it. Tracing what is
actually on the search path:

- **`FindMove` uses `Match` only via** `getBoardY()`, `getBoardX()`, `getMoveNr()`, and
  `setPin` / `removePin` / `getEval()` on those boards. **None of `Match`'s GUI code is reachable
  from `computeMove()`.**
- **`Match()`'s constructor is GUI-free**: it obtains the two boards, enables Zobrist, and wires
  itself into `FindMove`. Nothing Swing, `frame` untouched.
- `Match`'s GUI use is confined to the interactive flow: pie-rule dialogs (`JOptionPane`, lines
  321–339), a wait cursor (366–382) and a match-over message (427).

**Four constraints a driver would have to respect — each a place this could still fail:**

1. **`Match.prepareNewMatch(MatchData, boolean)` calls `RightPanel.getInstance()`** (line 139), as
   does `updateMatchData()` (line 153). These are the obvious way to start a configured game, and
   **both touch a GUI singleton.** A headless driver must set the position up without them.
2. **`Match()` and `prepareNewMatch` are package-private**, so **the driver must live in package
   `net.schwagereit.t1j`** (or use reflection).
3. **`Move.toString()` returns `GuiBoard.getHoleName(x, y, false)`** — so merely *logging a move*
   loads a Swing-importing class. An easy trap for an adapter.
4. **`GeneralSettings.correct()` references `GuiMainWindow.SCHEME_NUMBER`.** That field is
   `public static final int SCHEME_NUMBER = 8` — a compile-time constant, which `javac` inlines, so
   it should **not** trigger `GuiMainWindow` class initialization. **This is the single most
   load-bearing inference in this section and it rests on how the code was compiled, not on what it
   says. E2 must confirm it by running.**

**Assessment: a headless path plausibly exists, but it is narrow and none of it is confirmed.**
Feasibility is a static judgement here; only E2 can establish it. Criterion 7 stays `~`.

### Preferences: an E2 isolation requirement, not a deferred E3 concern

> **Corrected in review round 1.** This was first written as a determinism hazard for E3 to close.
> That was wrong about the direction of the risk: `node()` **writes**, so the exposure begins at
> the first class load in E2, not at measurement time in E3.

`GeneralSettings` is an **eagerly-initialised singleton**
(`private static final GeneralSettings ourInstance = new GeneralSettings();`) whose constructor
calls `Preferences.userRoot().node("/net/schwagereit/t1j")` and pulls `FixedPly`, `Ply`, `Time` and
`Colorscheme` from it. `FindMove.computeMove()` then reads `mdFixedPly` / `mdPly` / `mdTime` from
that singleton, and `MatchData` uses the same node.

**`node(pathName)` is not a read.** The API states it returns the named node *"creating it and any
of its ancestors if they do not already exist"*, and that nodes so created *"are not guaranteed to
become permanent until the `flush` method is called"* — **not guaranteed** is not the same as
**guaranteed not**, and the JDK's preferences implementation syncs on its own schedule. So merely
loading `GeneralSettings` can **create `/net/schwagereit/t1j` and its ancestors in the host user's
preference store**, and that state may be persisted without anything in T1j ever calling `flush`.
(`savePreferences()` writes outright, but nothing on the search path calls it.)

**Two consequences, in the right order:**

1. **E2 must isolate the preferences backing store before the first class load** — an isolated
   user home or an explicitly redirected backing store — and must **verify the host preference
   store is unchanged afterwards**. Deferring this to E3 is too late: by then the node may already
   exist on the machine. This is a side effect on the operator's system, not just an experimental
   nuisance.
2. **E3 then verifies that the configured values actually control the search** — that the ply or
   time in force is the one set, and not one recovered from a store. That is a separate question
   from isolation and it does not substitute for it.

The underlying determinism risk stands either way: the search depth used depends on what the
backing store holds, so a "fixed" setting could silently differ between machines or between runs
unless it is pinned and proven.

---

## What E1 did not do

No installation, no compilation, no class-file extraction, no `javap`, no class loading, no
execution of any Java code, no adapter, no games, no seeds. The clone and JAR were written only to
session scratch, never to this repository. Class bytecode was not examined, so **all rules and
control-flow findings above are properties of the source, not of the artifact that would play.**

## Carried into E2 — not authorized here

1. **Isolate the preferences backing store before the first class load**, and verify afterwards
   that the host user's preference store is unchanged. This is a prerequisite, not a step: the
   exposure begins the moment `GeneralSettings` is touched.
2. Confirm a current JVM loads the 2007 class files at all.
3. Confirm the headless path by running it, especially that `GuiMainWindow` is never initialised.
4. If a source build is ever preferred to the JAR, resolve `source/target="1.4"` first — while
   noting that a rebuild answers a different question than authenticating the released binary.

Carried into **E3**: verify the configured ply or time actually controls the search, and
behaviourally qualify the artifact over the corpus its games exercise — which is not the same as
authenticating its bytecode.
