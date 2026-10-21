# Version 3 (superseded)

The version 2 model with a familiarity check added (2026-10-20). It was served until version 4 replaced it.

- **Why it existed:** a phone test of version 2 showed portraits and landscapes labelled "paper" at 80% confidence. Version 3 kept the same trained weights and added a check that flags photos unlike the training photos instead of labelling them.
- **The model:** EfficientNet-B0 trained on TrashNet only (version 2's run `efficientnet_b0/seed44`, checkpoint SHA-256 `bfa1f3d3…`). It was re-exported to ONNX with a second output, the features the check reads (SHA-256 `155e0de2…`). The logits are identical to the version 2 file that was evaluated, on all 2,155 photos compared ([model.json](model.json)).
- **The familiarity check:** cosine similarity of a photo's features to its nearest training photo, among the 1,795 training photos (stored as 8-bit unit vectors). The threshold is 0.535, so 95% of validation photos pass.
  - False alarms on new waste photos: 5.2%.
  - Everyday Imagenette photos flagged: 91.6% of 2,425 ([familiarity.json](familiarity.json)).
- **Test (version 2's single evaluation; the weights did not change):** 366 TrashNet photos, accuracy 94.3% (95% CI 90.3–96.9%, resampling groups) and macro F1 0.936 ([test_metrics.json](test_metrics.json)).
  - The intervals were recomputed once from the recorded predictions, without running the model again. The first grouped bootstrap had scored a class missing from a resample as F1 0.
- **Latency:** 76 ms mean and 98 ms p95 per upload through Docker ([latency.json](latency.json)).

**What replaced it.** On real waste photographed on a landfill floor (RealWaste), version 3 was right on only 33.0% of the version 4 test photos, and its check flagged 78.8% of them as unfamiliar. Version 4 adds RealWaste to training and replaces the check with a Mahalanobis distance, which works better for the broader model. Version 4's evaluation scored version 3 on the same photos, from its release files, for the comparison in the [main README](../../../README.md#result-in-brief).

**Files kept here:**
- the records version 3 was served and documented with: [model.json](model.json), [model_selection.json](model_selection.json), [baselines.json](baselines.json), [split_stats.json](split_stats.json), [test_metrics.json](test_metrics.json), [val_errors.json](val_errors.json), [familiarity.json](familiarity.json), [latency.json](latency.json);
- its figures, in [figures/](figures/).

The model is still downloadable as GitHub release `model-v3`. Version 4's evaluation reads [model.json](model.json) to check the downloaded files' SHA-256. Its split is the TrashNet part of version 4's split, unchanged ([splits/trashnet-v2.csv](../../../splits/trashnet-v2.csv)).
