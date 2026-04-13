# Plan: Add Value-Head Caution Icons + Rolling "Value Instability" Warning

**Status: COMPLETED** - This plan was implemented in trainer.py

## Goal

At the end of each iteration's Sanity block, print:
- Nothing if healthy
- `⚠️` / `🚨` / `💥` if value head metrics are out of range
- An additional rolling-window warning if the issue persists across eligible iters

---

## 1. Define Thresholds (near other constants in trainer.py)

```python
# Value head "happy" bands (pre-tanh magnitude)
VALUE_P99_CAUTION = 4.0
VALUE_P99_WARN = 4.6
VALUE_P99_CRIT = 5.2
VALUE_SAT_CAUTION = 0.02
VALUE_SAT_WARN = 0.05
VALUE_SAT_CRIT = 0.15

# Rolling window for instability detection
VALUE_WINDOW = 20
VALUE_WARN_FRACTION = 0.50  # warn if >= 50% of eligible iters are warn-level
VALUE_MIN_ELIGIBLE = 10     # don't evaluate rolling logic until this many eligible iters

# Eligibility
VALUE_MIN_SAMPLES = 128     # minimum sanity sample count to be eligible
```

---

## 2. Add Tracking State (in trainer.py, near balance tripwire state)

```python
from collections import deque

# Value head instability tracking
value_history = deque(maxlen=VALUE_WINDOW)  # stores {eligible, level, trigger, p99, sat}
value_warn_streak = 0  # consecutive warn/crit eligible iters
value_instability_active = False  # for noise control: only print rolling warning once per streak
```

---

## 3. Compute Per-Iteration "Value Health Level"

Add helper function:

```python
def get_value_health_level(p99: float, frac_sat: float, n_samples: int) -> tuple[bool, str, str]:
    """
    Determine value head health level.

    Returns:
        (eligible, level, trigger) where:
        - level is one of: "ok", "caution", "warn", "crit"
        - trigger is one of: "p99", "sat", "both", "" (empty if ok/skip)
    """
    # Check eligibility - "skip" is never treated as warn/crit
    if n_samples < VALUE_MIN_SAMPLES or p99 is None or frac_sat is None:
        return False, "skip", ""

    # Determine level and trigger (check crit first, then warn, then caution)
    p99_crit = p99 >= VALUE_P99_CRIT
    sat_crit = frac_sat >= VALUE_SAT_CRIT
    if p99_crit or sat_crit:
        trigger = "both" if (p99_crit and sat_crit) else ("p99" if p99_crit else "sat")
        return True, "crit", trigger

    p99_warn = p99 >= VALUE_P99_WARN
    sat_warn = frac_sat >= VALUE_SAT_WARN
    if p99_warn or sat_warn:
        trigger = "both" if (p99_warn and sat_warn) else ("p99" if p99_warn else "sat")
        return True, "warn", trigger

    p99_caution = p99 >= VALUE_P99_CAUTION
    sat_caution = frac_sat >= VALUE_SAT_CAUTION
    if p99_caution or sat_caution:
        trigger = "both" if (p99_caution and sat_caution) else ("p99" if p99_caution else "sat")
        return True, "caution", trigger

    return True, "ok", ""
```

---

## 4. Print Single Concise Status Line (Icons)

Location: Immediately after the `v pretanh:` line in the Sanity block.

```python
# Get health level
eligible, level, trigger = get_value_health_level(p99_pretanh, frac_sat, n_samples)

# Print status (quiet when healthy or skipped - only print problems)
if not eligible:
    pass  # Don't print "skipped" - noise control
elif level == "ok":
    pass  # Print nothing when healthy
elif level == "caution":
    print(f"    ⚠️  Value head: p99={p99_pretanh:.2f}, sat={frac_sat:.3f} (caution; trigger={trigger})")
elif level == "warn":
    print(f"    🚨 Value head: p99={p99_pretanh:.2f}, sat={frac_sat:.3f} (warning; trigger={trigger})")
elif level == "crit":
    print(f"    💥 Value head: p99={p99_pretanh:.2f}, sat={frac_sat:.3f} (critical; trigger={trigger})")
```

---

## 5. Update Streak + History Every Iter

After computing level:

```python
# Record to history (normalize: level/trigger only meaningful when eligible)
value_history.append({
    "eligible": eligible,
    "level": level if eligible else None,
    "trigger": trigger if eligible else "",
    "p99": p99_pretanh if eligible else None,
    "sat": frac_sat if eligible else None,
})

# Update streak (only if eligible)
if eligible:
    if level in ("warn", "crit"):
        value_warn_streak += 1
    else:
        value_warn_streak = 0
```

---

## 6. Add Rolling-Window "VALUE INSTABILITY" Warning

After the per-iter icon line:

```python
# Compute rolling stats from eligible entries
eligible_entries = [x for x in value_history if x["eligible"]]
eligible_count = len(eligible_entries)

if eligible_count >= VALUE_MIN_ELIGIBLE:
    warn_only = sum(1 for x in eligible_entries if x["level"] == "warn")
    crit_only = sum(1 for x in eligible_entries if x["level"] == "crit")
    warn_crit_count = warn_only + crit_only
    warn_rate = warn_crit_count / eligible_count

    if warn_rate >= VALUE_WARN_FRACTION:
        # Only print once per streak (noise control)
        if not value_instability_active:
            value_instability_active = True
            print(f"    🚨 VALUE INSTABILITY: warn/crit in {warn_crit_count}/{eligible_count} eligible iters "
                  f"(warn={warn_only} crit={crit_only}, window={VALUE_WINDOW}, threshold={VALUE_WARN_FRACTION:.0%}) | "
                  f"consec_warn_streak={value_warn_streak}")
    else:
        value_instability_active = False  # reset when condition clears
# else: don't reset value_instability_active - avoid forgetting state during eligibility dips
```

---

## 7. Action Hints (only on warn/crit)

On first warn/crit of a streak (streak == 1):

```python
if eligible and level in ("warn", "crit") and value_warn_streak == 1:
    print(f"    Suggested: lower --value-lr-scale, reduce value_grad_max_norm, or increase sims")
```

On critical:

```python
if eligible and level == "crit":
    print(f"    Suggested: immediately lower value LR and/or clamp value grads")
```

---

## 8. Implementation Location

All changes go in `scripts/GPU/alphazero/trainer.py`:

1. **Constants** - near line ~67 (after SIMS_TABLE, MAX_MOVES_TABLE)
2. **State variables** - near line ~955 (after balance tripwire state)
3. **Helper function** - near line ~170 (after get_scaled_train_steps)
4. **Per-iter logic** - in the Sanity block, after `v pretanh:` line (~line 1270)

---

## 9. Example Output

**Healthy iteration (no output after v pretanh line):**
```
  Sanity (560 positions):
    z: mean=0.002, std=0.666, [+/0/-]=112/280/111
    z by to_move: red=0.444 (+/0/-=112/140/0), black=-0.442 (+/0/-=0/140/111)
    v: mean=-0.030, std=0.010, range=[-0.06,-0.01], mse=0.4715, sign_agree=50.0%
    v pretanh: range=[-0.06,-0.01], p99=0.05, frac_sat=0.000, zv_corr=0.006 (n=560)
```

**Caution iteration:**
```
    v pretanh: range=[-3.2,4.1], p99=4.31, frac_sat=0.018, zv_corr=0.123 (n=560)
    ⚠️  Value head: p99=4.31, sat=0.018 (caution; trigger=p99)
```

**Warning with instability (prints once when instability first detected):**
```
    v pretanh: range=[-4.8,5.1], p99=5.12, frac_sat=0.059, zv_corr=0.089 (n=560)
    🚨 Value head: p99=5.12, sat=0.059 (warning; trigger=sat)
    Suggested: lower --value-lr-scale, reduce value_grad_max_norm, or increase sims
    🚨 VALUE INSTABILITY: warn/crit in 12/20 eligible iters (warn=9 crit=3, window=20, threshold=50%) | consec_warn_streak=5
```

**Critical:**
```
    v pretanh: range=[-6.2,6.6], p99=6.61, frac_sat=0.371, zv_corr=0.034 (n=560)
    💥 Value head: p99=6.61, sat=0.371 (critical; trigger=both)
    Suggested: immediately lower value LR and/or clamp value grads
```
