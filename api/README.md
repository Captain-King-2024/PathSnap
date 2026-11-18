# api

FastAPI application, endpoints, model loading and server-side evaluation. Evaluator labels must never enter prediction features or UI responses.

Run the demo with `python -m api.server` from the repository root, then open `http://127.0.0.1:8000`. It uses the saved tuned HGB artifact and the processed student-test trips. See [training setup](../src/training/README.md) for preprocessing commands. The road-state HMM combines model probabilities with directed-road transition scores. Viterbi finds the best-scoring candidate sequence for each connected piece. Aligned student labels are used for scoring after matching.

For the FastAPI entry point, install `requirements-demo.txt` and run `python -m uvicorn api.app --host 127.0.0.1 --port 8000`. It serves the same UI and prediction service. For a second laptop on a trusted LAN, bind the server to `0.0.0.0` and open `http://HOST_IP:8000` from that laptop.

- `GET /api/trips` lists available trips and the active split.
- `GET /api/match/{trip}` returns GPS points, candidate roads, selected segments, route break indices, and available student scores.

The demo defaults to `student_test`. Use `--split evaluator_test` (or set `PATHSNAP_SPLIT` for FastAPI) when the evaluator inputs have been prepared as `data/processed/evaluator_test/<trip>/inputs.npz` in the same schema. Supply those inputs locally at evaluation time. For the FastAPI server, `PATHSNAP_DATA`, `PATHSNAP_LABELS`, and `PATHSNAP_MODEL` override their respective paths.
