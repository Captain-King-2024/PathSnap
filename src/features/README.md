# features

Candidate generation and feature extraction. Keep candidate-to-arc correspondence; fit learned transformations on training folds only.

Run from the repository root after preprocessing:

```powershell
python src/features/candidates.py
python -m unittest discover -s tests
```

The default candidate set contains up to eight directed arcs within 100 metres, ordered by exact point-to-segment distance and arc ID. If none fall within the radius, the nearest eight segments across the network are returned and that observation is flagged. Candidates are never forced to include reference arcs. The radius and cap are initial engineering defaults, not optimised parameters.

A uniform spatial grid indexes radius-expanded segment bounding boxes, so long segments crossing a point's cell remain discoverable. Projection onto segments is clamped to endpoints; zero-length arcs are handled. Reverse arcs retain separate IDs. Ties use arc IDs only for reproducibility, not as model features.

Features: distance, rank, projection fraction, segment length, directed heading cosine, heading-availability indicator, previous step length, previous time interval, previous speed, large-gap indicator, start-node out-degree and end-node in-degree. Heading is unavailable for the first observation, movements under two metres or gaps over five seconds; the indicator distinguishes this from a perpendicular heading. Speed is set to zero for a zero time interval and the time feature is retained. All numeric values must be finite. No learned scaling is performed.

Outputs: ignored data/features/{partition}/{trip}/candidates.npz and metadata.json. Offsets delimit candidate rows for each GPS observation; observation_ids preserve original GPS row indices. Candidate arc IDs and snapped coordinates are metadata, outside the feature matrix. Settings and input hashes are recorded. Audit reports go to results/features.

Only processed train and student_test inputs are read. Reference files are never opened. These outputs do not yet contain training targets: the chronological .route sequence requires a separately documented observation-alignment method. Candidate miss rate cannot be claimed until reference alignment exists. Candidate generation and feature extraction are complete; target alignment is the next step before supervised training.

