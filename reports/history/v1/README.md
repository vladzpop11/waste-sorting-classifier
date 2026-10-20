# Version 1 (superseded)

The first trained model and its one test evaluation (2026-10-20).

- **The split:** average-linkage clusters of ImageNet ResNet18 features, at cosine 0.86, within each class ([split.csv](split.csv)).
- **What a review found:** average linkage joins clusters by their mean similarity, so individual same-object pairs could still fall on opposite sides of the split.
  - 44 test photos had a same-class training photo at cosine ≥ 0.86, and 11 at ≥ 0.88. Some of them were visibly the same object.
  - The model scored 100% on those 44, and 95.0% on the other 317 test photos.
- **What changed:** the split was rebuilt so that no same-class pair at ≥ 0.86 straddles it. The models were retrained (three seeds per architecture), and the new test set was evaluated once.

Version 1 results, kept for the record:
- **Validation:** EfficientNet-B0 95.6% accuracy, 0.946 macro F1; ResNet18 93.1%, 0.915.
- **Test:** accuracy 95.6% (95% CI 93.4–97.5%), macro F1 0.945.
- **The model:** [model.json](model.json), ONNX SHA-256 `ac2385ef…`. It is still downloadable as GitHub release `model-v1`.
