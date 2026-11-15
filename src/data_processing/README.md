# data_processing

Read, inspect, clean and preprocess .track, .nodes, .arcs and .route files. Preserve trip-level partitions.

Run from the repository root:

```powershell
python -m pip install -r requirements-preprocessing.txt
python src/data_processing/preprocess.py --data-root "PATH_TO_STUDENT_FOLDER"
python -m unittest discover -s tests -p "test_preprocessing.py"
```

The input root must contain train (80 trips) and student_test (10 trips). Evaluator data is not processed. Outputs go to ignored data/processed; audit CSV and summary JSON go to results/preprocessing.

Each trip exports inputs.npz and metadata.json. Reference routes are saved separately under each partition's _references directory. Original node/arc IDs and GPS row IDs are preserved. Invalid GPS rows are removed with an audit count; invalid network IDs and decreasing timestamps fail validation. Duplicate timestamps, zero-length arcs and gaps are retained and reported for later feature/decoder handling.

Coordinates are projected into local equirectangular metres around the median network position. This is a local approximation; inspect extent_m before using it over very large regions. No learned scaler is fitted here. Fit learned preprocessing separately within training folds.

The .route file is a chronological arc sequence, not one label per GPS row. Pointwise reference alignment and candidate generation are subsequent tasks. Never zip route rows directly with GPS observations. Reference disconnections are reported rather than silently repaired. Components are weak/undirected connectivity counts, not directed reachability.

