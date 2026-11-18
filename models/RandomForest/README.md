# RandomForest Models

This directory is reserved for Random Forest artifacts during model exploration and evaluation:
- `model.joblib`: Serialized trained scikit-learn model
- `preprocessing.npz`: Fitted scaler mean and scale parameters
- `config.json`: Hyperparameters and configuration snapshot
- `feature_names.json`: Feature list used for training
- `training_summary.json`: CV metrics, OOB score, and feature importances

Once all candidate models are evaluated, the winning model architecture and artifacts will be promoted to the final production location.
