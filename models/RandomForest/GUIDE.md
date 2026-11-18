# PathSnap Random Forest — Developer Quick Reference

## Pipeline Overview

```
Raw data  →  preprocess.py  →  candidates.py  →  align_routes.py  →  train.py  →  model artifacts
```

---

## When to Re-Run What

| You changed… | Re-run these steps |
|---|---|
| Nothing — just re-training | Step 4 only |
| `configs/random_forest.json` params | Step 4 only |
| `src/features/candidates.py` (feature logic) | Steps 2 → 3 → 4 |
| `src/training/align_routes.py` (label logic) | Steps 3 → 4 |
| `src/data_processing/preprocess.py` | Steps 1 → 2 → 3 → 4 |
| Raw data changed | All steps (1 → 2 → 3 → 4) |

---

## Commands (run from `/home/hecker/Documents/Hackathon/`)

### Step 1 — Preprocessing (only if raw data changed)
```bash
/home/hecker/jupyter-env/bin/python src/data_processing/preprocess.py \
    --data-root "/home/hecker/Downloads/Student folder" \
    --output data/processed \
    --report-dir results/preprocessing
```
✅ Done when: `results/preprocessing/summary.json` shows `"successful_trips": 90`

---

### Step 2 — Feature Extraction (only if preprocessing or candidates.py changed)
```bash
/home/hecker/jupyter-env/bin/python src/features/candidates.py \
    --input-root data/processed \
    --output data/features \
    --report-dir results/features
```
✅ Done when: `results/features/summary.json` shows `"trips": 90`

---

### Step 3 — Label Alignment (only if align_routes.py changed or features changed)
```bash
/home/hecker/jupyter-env/bin/python src/training/align_routes.py \
    --inputs data/processed \
    --output data/labels
```
✅ Done when: prints `train {"trips": 80 ...}` and `student_test {"trips": 10 ...}`

---

### Step 4 — Train Random Forest ← **this is the main one you'll run**
```bash
/home/hecker/jupyter-env/bin/python src/training/train.py \
    --features data/features \
    --labels data/labels \
    --output results/training \
    --models forest \
    --folds 5 \
    --jobs 4
```
✅ Done when: prints `forest student_test {...}` with accuracy numbers

Results saved to: `results/training/forest.json` and `results/training/forest.pkl`

---

### Step 5 — Optional: Hyperparameter Tuning
```bash
PYTHONPATH=. /home/hecker/jupyter-env/bin/python src/optimization/tune_random_forest.py \
    --n-trials 50
```

---

### Step 6 — Run All Tests (always do this before committing models)
```bash
cd /home/hecker/Documents/Hackathon && \
/home/hecker/jupyter-env/bin/python -m unittest discover -s tests -v
```

---

## Committing Model Artifacts Only (not code)

After a training run you're happy with, copy the artifacts and commit just those:

### 1. Copy artifacts to `models/RandomForest/`
```bash
cp results/training/forest.pkl   models/RandomForest/model.pkl
cp results/training/forest.json  models/RandomForest/training_summary.json
```

### 2. Stage ONLY the model folder
```bash
git add models/RandomForest/
```

### 3. Verify only model files are staged (no source code!)
```bash
git status
# Should show only:  models/RandomForest/*
# Nothing under src/, configs/, tests/, etc.
```

### 4. Commit on the RandomForest branch (never main)
```bash
git commit -m "feat: add Random Forest model artifacts (top1=81.72%, macroF1=89.36%)"
```

### 5. Double-check you are NOT on main
```bash
git branch
# Should show:  * RandomForest
```

---

## Results Reference (current best run)

| Metric | Random Forest | Logistic (baseline) |
|---|---|---|
| CV top-1 accuracy | 78.39% | 77.20% |
| CV macro F1 | 87.63% | 86.96% |
| **Student-test top-1** | **81.72%** | **80.48%** |
| **Student-test macro F1** | **89.36%** | **88.63%** |
| Binary log-loss | **0.1349** | 0.1845 |

---

## Full Pipeline (one-liner — only if starting from scratch)
```bash
cd /home/hecker/Documents/Hackathon && \
/home/hecker/jupyter-env/bin/python src/data_processing/preprocess.py --data-root "/home/hecker/Downloads/Student folder" --output data/processed --report-dir results/preprocessing && \
/home/hecker/jupyter-env/bin/python src/features/candidates.py --input-root data/processed --output data/features --report-dir results/features && \
/home/hecker/jupyter-env/bin/python src/training/align_routes.py --inputs data/processed --output data/labels && \
/home/hecker/jupyter-env/bin/python src/training/train.py --features data/features --labels data/labels --output results/training --models forest --folds 5 --jobs 4
```
