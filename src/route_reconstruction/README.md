# route_reconstruction

HMM emissions, road-network transitions and Viterbi decoding. Handle GPS gaps, missing candidates and disconnected roads.

The decoder treats each observation's candidate road as a state and uses the logistic model's candidate score as its emission weight. Transition weights penalize the difference between GPS displacement and bounded directed road-network distance. The decoder starts a new piece after intervals over 15 seconds, nonpositive intervals, or a step without a feasible candidate transition. These thresholds are initial defaults.

Run `python -m src.route_reconstruction.evaluate` to reproduce the comparison in `results/metrics/logistic_decoder.json`. Top-1 uses aligned observation labels and measures agreement at the observation level. The map displays selected road segments, with intermediate graph edges used for scoring transitions.
