# Correct Fix: --load-weights = weights-only, ignore JSON

## Goal

Let benchmarks (and any "fresh but strong policy" runs) load only the `.safetensors` weights without restoring:
- iteration
- curriculum active size
- freeze state / reduction factor
- promotable/demotable counters

So `--iterations 1` always runs, and `--curriculum-sizes` always wins.

---

## 1) scripts/GPU/alphazero/train.py - add flag + mutual exclusion + pass-through

### A) Add CLI arg (near where --resume is defined)

```python
parser.add_argument(
    "--load-weights",
    type=str,
    default=None,
    help="Load network weights only (no training state restore).",
)
```

### B) Mutual exclusion (right after args parsed)

```python
if args.resume and args.load_weights:
    parser.error("Cannot use both --resume and --load-weights")
```

### C) Pass into trainer (where you already pass resume_from=args.resume)

```python
trainer.train(
    ...,
    resume_from=args.resume,
    load_weights_from=args.load_weights,
)
```

Key: the name must match what you add in trainer.py: `load_weights_from`.

---

## 2) scripts/GPU/alphazero/trainer.py - split the resume block into 2 paths

Current resume/load block is around lines 1287-1315.

### A) Update the trainer entry signature to accept the new param

Wherever the function currently accepts `resume_from`, add:
- `load_weights_from: str | None = None`

Example:

```python
def train(..., resume_from=None, load_weights_from=None, ...):
    ...
```

### B) Replace current block with this exact structure

Replace:

```python
# Resume from checkpoint if specified
if resume_from:
    network.load_weights(resume_from)
    state_path = Path(resume_from).with_suffix(".json")
    ...
```

With:

```python
# Load weights-only if specified (no state restore)
if load_weights_from:
    network.load_weights(load_weights_from)
    print(f"Loaded weights-only from {load_weights_from} (no state restored)")

# Resume from checkpoint if specified (full state restore)
elif resume_from:
    network.load_weights(resume_from)
    state_path = Path(resume_from).with_suffix(".json")
    if state_path.exists():
        with open(state_path) as f:
            state = json.load(f)
            start_iteration = state.get("iteration", 0)

            # Restore curriculum state
            if "curriculum" in state:
                curriculum = CurriculumManager.from_dict(state["curriculum"])

            # Restore freeze state (critical for resume consistency)
            if "freeze_state" in state:
                fs = state["freeze_state"]
                consecutive_high_timeout_iters = fs.get("consecutive_high_timeout_iters", 0)
                consecutive_good_timeout_iters = fs.get("consecutive_good_timeout_iters", 0)
                curriculum_frozen = fs.get("curriculum_frozen", False)
                sims_reduction_factor = fs.get("sims_reduction_factor", 1.0)
                consecutive_saturation_iters = fs.get("consecutive_saturation_iters", 0)
                last_active_size = curriculum.active_size

            # Restore curriculum state (promotion/demotion tracking)
            if "curriculum_state" in state:
                cs = state["curriculum_state"]
                consecutive_promotable_iters = cs.get("consecutive_promotable_iters", 0)
                consecutive_demotable_iters = cs.get("consecutive_demotable_iters", 0)

    print(f"Resumed from {resume_from}, iteration {start_iteration}")
    print(f"  Curriculum: active_size={curriculum.active_size}")
    print(f"  Freeze: frozen={curriculum_frozen}, factor={sims_reduction_factor:.2f}")
    print(f"  Curriculum: promotable={consecutive_promotable_iters}, demotable={consecutive_demotable_iters}")
```

That's it. Weights-only must do ONLY `network.load_weights()` and a print. No JSON. No touching start_iteration, curriculum, freeze vars, counters.

---

## 3) scripts/bench_knee.py - switch to --load-weights

### A) Add bench CLI arg

```python
parser.add_argument(
    "--load-weights",
    type=str,
    default=None,
    help="Weights-only load (no training state restore) - passed to train.",
)
```

### B) Add passthrough in build_train_cmd()

```python
if load_weights:
    cmd.extend(["--load-weights", load_weights])
```

### C) For trained-policy benchmark use case: do not pass --resume

(You can keep --resume support for other reasons, but bench should not append both.)

---

## Verification (exactly what you wrote)

### Test 1 - weights-only standalone

```bash
python -m scripts.GPU.alphazero.train \
  --iterations 1 --train-steps 0 \
  --games-per-iter 2 --curriculum-sizes 16 --simulations 50 \
  --checkpoint-dir /tmp/bench_test \
  --load-weights checkpoints/alphazero-fresh/model_iter_0140.safetensors
```

Expected:
- Prints `Loaded weights-only ...`
- Runs iteration (doesn't exit immediately)
- Uses curriculum 16 (doesn't jump to 24)
- No "Resumed from ... iteration 140"

### Test 2 - bench knee with trained weights

```bash
python scripts/bench_knee.py --board 16 --sims 200 --games 12 --repeats 2 \
  --load-weights checkpoints/alphazero-fresh/model_iter_0140.safetensors
```

Expected:
- Normal throughput
- No FAILED ()
- No "No valid results"

---

## Files to Modify

1. **scripts/GPU/alphazero/train.py** - Add `--load-weights` arg + mutual exclusion + pass to trainer
2. **scripts/GPU/alphazero/trainer.py** - Add `load_weights_from` param, weights-only path before resume path
3. **scripts/bench_knee.py** - Add `--load-weights` passthrough, use instead of `--resume`

---

## Status: COMPLETED

This feature was implemented and verified working.
