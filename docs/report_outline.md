# Technical report outline

1. Dataset understanding and EDA: preprocessing, missing values, feature choices, trajectory/network plots and Folium map; identify partitions.
2. Every model: suitability, strengths/limitations, configuration, loss/objective, student-test results and over/underfitting evidence.
3. Overall comparison: CV macro F1, student-test top-1 accuracy, macro F1, log loss, route continuity and inference time. Include actual comparison plots.
4. Bayesian optimisation: search space, objective, group-aware folds, trial count, best parameters, computational cost, tuned/untuned comparison.
5. HMM/Viterbi: states, emissions, transitions, missing candidates, GPS gaps, disconnected roads and classifier-only/decoded comparison.
6. Folium: GPS, road geometry, development reference route, predicted route, labels, legend, layers and representative screenshot.
7. API/UI/deployment: working UI and successful receiving-laptop API evidence.
8. Limitations, reproducibility and AI assistance disclosure.

Save one consolidated PDF under reports/. Do not invent results or training curves.

