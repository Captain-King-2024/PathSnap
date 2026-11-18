# Bayesian hyperparameter optimisation

Install the requirements and run from the repository root after generating candidate features and aligned labels.

```powershell
python -m pip install -r requirements-optimization.txt
python -m src.optimization.tune --trials 30
```

The script tunes Histogram Gradient Boosting only. Use `--objective top1_accuracy` to select by observation-level top-1. The default objective is macro F1. Choose the objective before looking at the student-test results.

Optuna's TPE sampler starts with the runner's Histogram Gradient Boosting baseline and varies boosting iterations, maximum leaves per tree, learning rate, minimum samples per leaf, and L2 regularization. It uses three trip-grouped folds from the 80 training trips, then refits the baseline and winning settings on all 80 trips and compares them on the 10 student-test trips. The report goes to `results/optimization/boosting_bayesian.json` and the tuned model goes to `models/boosting_bayesian.pkl`.
