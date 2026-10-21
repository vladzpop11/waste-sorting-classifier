# Model card: waste sorting classifier, version 4

One page on what the model is for, what it learned from, how well it does, and where it fails. The evidence for every number is in the [README](README.md) and [reports/](reports/).

## The model

- **Task:** a photo of a single waste item → one of six classes: cardboard, glass, metal, paper, plastic, trash. A separate familiarity check flags photos unlike the training photos.
- **Architecture:** EfficientNet-B0, ImageNet-pretrained, fine-tuned end to end; input 224×224 RGB.
- **Files** ([release model-v4](https://github.com/vladzpop11/waste-sorting-classifier/releases/tag/model-v4)), each checked against its SHA-256 before use:
  - `model.onnx` (opset 17; `536ae512…`), with two outputs: logits and the 1,280 features;
  - `familiarity.bin` (`b15bff10…`);
  - `model.pt`, the training checkpoint (`dac7cf94…`).
- **Served** from Docker (ONNX Runtime and NumPy, no PyTorch) and in the browser ([live demo](https://huggingface.co/spaces/VladZpp/waste-sorting-classifier)). In the browser, photos stay on the device.

## Intended use

- **Suitable for:** a demonstration and teaching tool for sorting one waste item, photographed from above on a plain background or floor, into six recycling categories.
- **Not suitable for:**
  - automated sorting where a mistake has a cost, such as hazardous items, sharp objects or medical waste (none of these are classes);
  - photos of hands holding items, full bins, piles or busy scenes: both datasets lack them;
  - commercial use, which the data licence forbids (below).

## Training data

| | TrashNet | RealWaste |
| --- | --- | --- |
| Photos | 2,521 | 3,904 (food organics and vegetation are not used; miscellaneous and textile trash → trash) |
| What they show | one item on a white background, studio-like | real waste on a concrete floor at one Australian landfill |
| Train / validation / test | 1,795 / 360 / 366 | 3,109 / 398 / 397 |
| Licence | MIT | CC BY-NC-SA 4.0 |

Photos of the same object are kept on one side of the split. No validation or test photo has a same-class training photo at cosine similarity 0.86 or above (ImageNet ResNet18 features); `make split` checks this ([reports/data.md](reports/data.md)).

## Results

The test set was evaluated once. Version 3 (trained on studio photos only) was scored on the same photos in the same run. The intervals are 95% and resample groups of similar photos.

| Test photos | Accuracy | Macro F1 | Version 3 accuracy | Difference |
| --- | --- | --- | --- | --- |
| RealWaste (397) | 88.7% (83.7–92.9%) | 0.883 | 33.0% | +55.7 points (+47.4 to +63.2) |
| TrashNet (366) | 94.8% (91.4–97.3%) | 0.930 | 94.3% | +0.5 points (−1.8 to +3.1) |
| Both (763) | 91.6% (88.1–94.3%) | 0.916 | | |

- **Weakest classes:** studio trash, F1 0.800, worse than version 3's 0.889 (only 19 test photos); and real-waste metal, F1 0.807, with the lowest recall (73%): crushed cans and foil are called plastic or trash.
- **Confidence:** the model is underconfident (expected calibration error 0.060). Answers at 90% confidence or more were right 99.1% of the time on test (446 photos).
- **Unfamiliar photos are harder:** test photos far from every training photo of their class score lower, for example 79.3% on RealWaste photos whose nearest training photo is at similarity 0.75–0.80.

## Familiarity check

- **The method:** the Mahalanobis distance from a photo's features to the nearest class mean of the training features. The threshold lets 95% of validation photos pass.
- **Measured** ([reports/familiarity.json](reports/familiarity.json)):
  - False alarms on new waste photos: 4.9%.
  - Everyday photos (2,425 held-out Imagenette photos) flagged: 83.8%; 10.3% pass with 70% confidence or more.
  - On test, it flagged 10.6% of RealWaste and 6.8% of TrashNet photos.
- **It is not a guarantee:** about one everyday photo in six still gets a label.

## Limitations

- **Two datasets, one setting each.** Accuracy on other places, lighting or cameras is unknown.
- **Small test sets:** accuracies are uncertain by about ±3–4 points, and some classes have few photos.
- **The TrashNet test set has been used twice** (versions 2 and 4). Nothing in version 4 was chosen on it.
- **The split guarantee stops at 0.86 similarity.** Below it, some test objects may also appear in training, which would make scores slightly optimistic.

## Licence

The weights are trained on RealWaste, so they are released under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). That means non-commercial use only, attribution to both datasets, and adaptations under the same licence. Please cite:
- TrashNet: Thung & Yang, 2016.
- RealWaste: Single, Iranmanesh & Raad, 2023, [doi:10.3390/info14120633](https://doi.org/10.3390/info14120633).

The code is MIT ([LICENSE](LICENSE)).
