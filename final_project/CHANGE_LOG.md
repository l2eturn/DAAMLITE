# Change Log

Date: 2026-04-29

---

## Bug Fixes

### Fix 1 — `model_comparison.py:226` — Wrong fold count in plot title

**File**: `nyc-taxi-fare-predic/model_comparison.py`
**Line**: 226 (inside `plot_results()`)

**Problem**: The CV RMSE bar chart title used `len(names[0].split())` to determine the fold count. For the first model name `"Linear Reg."`, `"Linear Reg.".split()` produces `["Linear", "Reg."]` (length 2), so the chart was always titled `"2-fold CV RMSE Comparison"` regardless of the actual `N_FOLDS` value (5).

**Before**:
```python
ax1.set_title(f"{len(names[0].split())}-fold CV RMSE Comparison",
```

**After**:
```python
ax1.set_title(f"{N_FOLDS}-fold CV RMSE Comparison",
```

**Impact**: Visual/labeling bug — chart title was always `"2-fold"` instead of `"5-fold"`.

---

### Fix 2 — `model_comparison.py` — Missing `best_model.txt` export breaks pipeline

**File**: `nyc-taxi-fare-predic/model_comparison.py`
**Location**: `main()` function, end of file

**Problem**: `tuning.py` (Phase 4) reads `best_model.txt` at startup via:
```python
best = open("best_model.txt").read().strip()
```
However, `model_comparison.py` (Phase 2) never writes this file. Running Phase 4 after a fresh Phase 2 would raise a `FileNotFoundError`. The file currently exists only because it was created manually.

**Fix**: Added `best_model.txt` export to `model_comparison.py`'s `main()`, immediately after saving `best_predictions.npy`:

```python
with open("best_model.txt", "w") as f:
    f.write(best_name)
print(f"  💾 Export: best_model.txt  ({best_name})")
```

**Impact**: Pipeline break — Phase 4 would crash with `FileNotFoundError` on a fresh run without the manually created file.

---

### Fix 3 — `submission.py:34` — Duplicate and incorrect `TEST_PATH` definition

**File**: `nyc-taxi-fare-predic/submission.py`
**Line**: 34 (removed)

**Problem**: Two module-level assignments to `TEST_PATH` existed:
- Line 34: `TEST_PATH = "test_taxi.csv"` — **incorrect** relative path (file does not exist there)
- Line 323: `TEST_PATH = "../raw-data/test_taxi.csv"` — **correct** path

Python uses the last assignment, so the script happened to work. However, the incorrect definition at line 34 was misleading and would cause a `FileNotFoundError` if someone reordered the module-level code or called `load_all()` from a different context before line 323 executed.

**Fix**: Removed the incorrect line 34 definition. The correct `TEST_PATH = "../raw-data/test_taxi.csv"` at line 323 (now line 321) remains as the sole definition.

**Before**:
```python
RANDOM_STATE = 42
N_FOLDS      = 5
TEST_PATH    = "test_taxi.csv"   # ← แก้ถ้าจำเป็น
```

**After**:
```python
RANDOM_STATE = 42
N_FOLDS      = 5
```

**Impact**: Dead code / latent bug — could cause `FileNotFoundError` if module load order changed.

---

## Other Observations (Not Fixed)

### `train_compare.py` is an exact duplicate of `model_comparison.py`

Both files are byte-for-byte identical. The project comment in `submission.py` references `phase3_comparison.py`, suggesting `train_compare.py` was intended to be a separate Phase 3 comparison script but was never differentiated from Phase 2. No fix applied since it is non-breaking, but it may be worth either deleting it or making it a distinct Phase 3 analysis.

### `submission.py` — Hardcoded hyperparameters may drift from tuning output

`get_best_tuned_model()` uses hardcoded parameter values (e.g., `Lasso(alpha=5.67)`). If `tuning.py` is re-run, the optimal parameters may change but `submission.py` will still use the old hardcoded values. Consider loading best params programmatically from `tuning_results.csv` or a serialized model file (e.g., `joblib.dump`).

---

## Files Changed

| File | Change |
|---|---|
| `nyc-taxi-fare-predic/model_comparison.py` | Fixed plot title fold count; added `best_model.txt` export |
| `nyc-taxi-fare-predic/submission.py` | Removed duplicate incorrect `TEST_PATH` definition |

## Files Created

| File | Description |
|---|---|
| `PROJECT_SUMMARY.md` | Full project summary in Markdown |
| `CHANGE_LOG.md` | This file |
