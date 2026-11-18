# Bayesian hyperparameter optimisation

Install the requirements and run from the repository root after generating candidate features and aligned labels.

```powershell
python -m pip install -r requirements-optimization.txt
python -m src.optimization.tune --model boosting --trials 30
```

Use `--model tree` to tune the standalone Decision Tree instead, or `--objective top1_accuracy` to select by observation-level top-1 rather than macro F1. Choose the model and objective before looking at the student-test results.

Optuna's TPE sampler starts with the runner's baseline settings and uses three trip-grouped folds from the 80 training trips. The runner refits the selected settings on all 80 trips and compares both settings on the 10 student-test trips. It writes the trial history and metrics to `results/optimization/<model>_bayesian.json` and the tuned model to `models/<model>_bayesian.pkl`. Existing baseline artifacts use separate filenames.
