Add a real trained model here as `cloud_classifier_model.keras` or `cloud_classifier_model.h5`. The web app does not include trained weights and will not fabricate predictions.

The model must accept `(150, 150, 3)` RGB images normalized to `[0, 1]` and return 11 scores. Verify that `CLASS_NAMES` exactly matches the training class-index order before using predictions. Do not share model weights without permission.
