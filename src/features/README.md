# features

Candidate generation and feature extraction. Keep candidate-to-arc correspondence; fit learned transformations on training folds only.

Run from the repository root after preprocessing:

```powershell
python src/features/candidates.py
python -m unittest discover -s tests
```

The default candidate set contains up to eight directed arcs within 100 metres, ordered by exact point-to-segment distance and arc ID. If none fall within the radius, the nearest eight segments across the network are returned and that observation is flagged. Candidates are never forced to include reference arcs. The radius and cap are initial engineering defaults, not optimised parameters.

A uniform spatial grid indexes radius-expanded segment bounding boxes, so long segments crossing a point's cell remain discoverable. Projection onto segments is clamped to endpoints; zero-length arcs are handled. Reverse arcs retain separate IDs. Ties use arc IDs only for reproducibility, not as model features.

Features: distance, rank, projection fraction, segment length, directed heading cosine, heading-availability indicator, previous step length, previous time interval, previous speed, large-gap indicator, start-node out-degree and end-node in-degree. Heading is unavailable for the first observation, nonpositive time intervals, movements under two metres, gaps over five seconds or zero-length arcs. The indicator distinguishes unavailable headings from perpendicular headings. Speed is set to zero for a zero time interval and the time feature is retained. All numeric values must be finite. No learned scaling is performed.

Outputs: ignored data/features/{partition}/{trip}/candidates.npz and metadata.json. Offsets delimit candidate rows for each GPS observation; observation_ids preserve original GPS row indices. Candidate arc IDs and snapped coordinates are metadata, outside the feature matrix. Settings and input hashes are recorded. Audit reports go to results/features.

Only processed train and student_test inputs are read during generation. The chronological .route sequence requires observation alignment before it can supply pointwise training targets.

A verification run independently recalculated all 12 features for 1,500,896 saved candidate rows. It compared the indexed search with exhaustive segment search for 3,681 observations across all 90 trips, including every fallback. All candidates and their ordering matched at the tested observations.

In a separate diagnostic, at least one arc from anywhere on the trip's reference route appeared among the candidates for 186,820 of 208,417 training observations and 19,132 of 19,160 student-testing observations. These fractions are upper bounds on pointwise candidate coverage against labels drawn from the supplied routes. They do not establish which reference arc belongs to any given observation. No zero-length segments appeared among the saved candidates.

