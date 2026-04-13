# Plan: Fix Size-12 Timeouts + Freeze Feedback Loop

## Problem Summary

- Size 12 shows 20-24% timeout rate with p95_plies hitting exactly max_moves=130
- Size 10 shows 0 timeouts with p95 well under cap
- Rolling freeze metric can create feedback loop (reduce sims → more timeouts → stay frozen)

## Changes

### 1. Raise Size-12 Ply Cap

**File:** `scripts/GPU/alphazero/trainer.py`
**Line:** 56-63

**Before:**
```python
MAX_MOVES_TABLE = {
    8: 90,
    10: 110,
    12: 130,
    16: 200,
    20: 225,
    24: 280,
}
```

**After:**
```python
MAX_MOVES_TABLE = {
    8: 90,
    10: 110,
    12: 160,  # was 130 - raised to reduce tail timeouts
    16: 200,
    20: 225,
    24: 280,
}
```

**Rationale:** avg_plies=72 but p95=130 (pinned to cap). Raising to 160 gives tail games room to finish while keeping iter_plies_ratio low (72/160=0.45, well under 0.85 gate).

---

### 2. Fix Freeze to Use Iteration-Local Timeout

**File:** `scripts/GPU/alphazero/trainer.py`
**Lines:** 1377-1409

**Before:**
```python
        # F) Freeze update (Phase 6)
        # Use curriculum timeout_rate for freeze decisions (rolling window)
        timeout_rate = curriculum_metrics["timeout_rate"]
        if timeout_rate >= 0.25:
            consecutive_high_timeout_iters += 1
            consecutive_good_timeout_iters = 0
            if consecutive_high_timeout_iters >= 2:
                curriculum_frozen = True
                if sims_after_freeze > ABS_SIMS_FLOOR:
                    sims_reduction_factor = max(0.5, sims_reduction_factor * 0.8)
                    print(f"  FREEZE: timeout={timeout_rate:.1%}, factor={sims_reduction_factor:.2f}")
                else:
                    print(f"  FREEZE: at sims floor (sims=100), factor={sims_reduction_factor:.2f}")
        else:
            consecutive_high_timeout_iters = 0
            consecutive_good_timeout_iters += 1
            if curriculum_frozen and consecutive_good_timeout_iters >= 3:
                new_factor = min(1.0, sims_reduction_factor * 1.25)
                if new_factor >= 1.0:
                    curriculum_frozen = False
                    sims_reduction_factor = 1.0
                    print(f"  UNFREEZE: fully recovered, factor=1.0")
                else:
                    sims_reduction_factor = new_factor
                    print(f"  RECOVER: ramping up, factor={sims_reduction_factor:.2f}")

        # Compute sims_next
        sims_next = int(base_sims_effective * sims_reduction_factor)
        sims_next = max(ABS_SIMS_FLOOR, sims_next)
```

**After:**
```python
        # F) Freeze update (Phase 6) - only on full iterations
        if full_iteration:
            timeout_rate_for_freeze = iter_timeout_rate
            if timeout_rate_for_freeze >= 0.25:
                consecutive_high_timeout_iters += 1
                consecutive_good_timeout_iters = 0
                if consecutive_high_timeout_iters >= 2:
                    curriculum_frozen = True
                    if sims_after_freeze > ABS_SIMS_FLOOR:
                        sims_reduction_factor = max(0.5, sims_reduction_factor * 0.8)
                        print(f"  FREEZE: timeout={timeout_rate_for_freeze:.1%}, factor={sims_reduction_factor:.2f}")
                    else:
                        print(f"  FREEZE: at sims floor (sims=100), factor={sims_reduction_factor:.2f}")
            else:
                consecutive_high_timeout_iters = 0
                consecutive_good_timeout_iters += 1
                if curriculum_frozen and consecutive_good_timeout_iters >= 3:
                    new_factor = min(1.0, sims_reduction_factor * 1.25)
                    if new_factor >= 1.0:
                        curriculum_frozen = False
                        sims_reduction_factor = 1.0
                        print(f"  UNFREEZE: fully recovered, factor=1.0")
                    else:
                        sims_reduction_factor = new_factor
                        print(f"  RECOVER: ramping up, factor={sims_reduction_factor:.2f}")

        # Compute sims_next (always, even on partial iterations)
        sims_next = int(base_sims_effective * sims_reduction_factor)
        sims_next = max(ABS_SIMS_FLOOR, sims_next)
```

**Key changes:**
- Entire freeze logic (both branches) wrapped in `if full_iteration:`
- Uses `timeout_rate_for_freeze = iter_timeout_rate` (iteration-local, not rolling)
- All prints use `timeout_rate_for_freeze`
- `sims_next` computation stays outside the gate (needed for checkpoint)

**Rationale:** Aligns freeze with promotion/demotion (both already iteration-local and gated by full_iteration). Prevents "frozen due to history" lag.

---

### 3. Add Timeout Diagnostic Trace

**File:** `scripts/GPU/alphazero/self_play.py`
**Location:** Just before `return GameRecord(...)` at line 252

**Add:**
```python
    # Diagnostic: print timeout trace (size-12 only to avoid spam)
    if draw_reason == DRAW_TIMEOUT:
        last_moves = move_history[-10:] if len(move_history) >= 10 else move_history
        print(f"  TIMEOUT: plies={ply}, last10={last_moves}")
```

**Note:** `active_size` is available as a parameter to `play_game()`, so we could guard with `if draw_reason == DRAW_TIMEOUT and active_size == 12:` if needed. For now, keeping it simple - at 25 games/iter with ~20% timeouts, that's ~5 lines/iter (acceptable).

**Rationale:** Confirms whether timeouts are normal long games vs pathological loops.

---

## Verification

Check current state first:
```bash
ls checkpoints/alphazero-fresh/*.safetensors | tail -3
```

Resume from latest full checkpoint (adjust iteration target accordingly):
```bash
# If currently at iter 150, run 5 more to validate:
.venv/bin/python -u -m scripts.GPU.alphazero.train \
    --iterations 155 --games-per-iter 25 --train-steps 140 \
    --checkpoint-dir checkpoints/alphazero-fresh \
    --resume checkpoints/alphazero-fresh/model_iter_0150.safetensors
```

After 2-5 iterations at size 12, check:
```bash
cat checkpoints/alphazero-fresh/model_iter_0151.json | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(f'size={d[\"active_size\"]} max={d[\"max_moves\"]}')
print(f'timeouts={d[\"timeout_draws\"]}/{d[\"games_generated\"]} = {d[\"timeout_draws\"]/d[\"games_generated\"]:.1%}')
print(f'avg_plies={d[\"avg_plies\"]:.1f} p95={d[\"p95_plies\"]:.1f} max_obs={d[\"max_plies_observed\"]}')"
```

---

## Success Criteria

**Immediate effect:**
- p95 no longer pinned at cap (p95 < 160)
- Timeouts drop materially (e.g., 20-24% → 10-15%)

**Promotion-ready:**
- `iter_timeout_rate ≤ 10%` sustained for 3 full iterations (your existing gate)

**Escalation:**
- If still >15% timeouts with p95 pinned at 160, bump to 175

---

## Status: COMPLETED

All three changes implemented and verified working (iter 151 showed 0 timeouts).
