# Data: duplicates, repeated objects and the split

Generated numbers come from `make split` ([split_stats.json](split_stats.json)) and `python -m waste_sorting_classifier.data_report` ([data_stats.json](data_stats.json), [figures/similarity_pairs.png](figures/similarity_pairs.png)).

## 1. The same photo under two labels

TrashNet has 2,527 photos. Three photos appear twice, byte for byte, filed under different classes:

- glass115 / metal91
- glass176 / plastic152
- glass389 / plastic332

Their true label is unknowable, so all six files are dropped: 2,521 photos remain. No byte-identical copies remain within a class either.

## 2. The same object photographed several times

TrashNet often photographs one object several times, from different angles. A random split would put photos of one object in both training and test, and the test score would then partly measure recognising familiar objects.

**How similar photos are found.** Each photo is described by the 512 features of an ImageNet-pretrained ResNet18 (not the trained model), and compared by cosine similarity within its class.

**Where the threshold sits.** [figures/similarity_pairs.png](figures/similarity_pairs.png) shows random same-class pairs from each similarity band:

| Cosine similarity | Same-class pairs | What the pairs are |
| --- | --- | --- |
| ≥ 0.95 | 41 | Near-identical photos of one object |
| 0.90–0.95 | 303 | The same object, re-posed |
| 0.88–0.90 | 351 | Almost always the same object |
| 0.86–0.88 | 719 | Mostly the same object, plus some look-alikes of one product type |
| 0.84–0.86 | 1,449 | Mixed: similar objects, sometimes the same one |
| 0.80–0.84 | 6,901 | Different objects of the same kind |

**The threshold is 0.86.** Going lower is not practical: joining pairs below about 0.85 chains look-alikes together into groups that cover half a class, and the classes could then no longer be split evenly.

**The grouping.** Within each class, every pair at ≥ 0.86 is joined (connected components), and average-linkage clusters at 0.86 are merged in. Joining every pair is what makes the guarantee below hold. Average linkage alone does not: version 1 of this split used only that, and it leaked (below).

**The split.** Groups are assigned with StratifiedGroupKFold (7 folds: one test, one validation, five train): 1,795 train, 360 validation and 366 test photos, with each class at 14–16% in test. `make split` refuses to save a split where any validation or test photo has a same-class training photo at ≥ 0.86. The highest such similarity is 0.8599 for validation and 0.8596 for test.

## 3. What remains

The guarantee stops at the threshold. Just below it, validation and test photos still have similar training photos, and some of those are probably the same object:

| Nearest same-class training photo | Validation (360) | Test (366) |
| --- | --- | --- |
| ≥ 0.85 | 29 | 46 |
| ≥ 0.84 | 74 | 100 |
| ≥ 0.80 | 241 | 269 |

The test score could therefore still be slightly optimistic for objects photographed many times. The README reports test accuracy by similarity to the nearest training photo: it is 97.8% for photos with a training neighbour at ≥ 0.80, and lower for photos further from anything in training.

**One very large group.** Joining every pair at ≥ 0.86 also chains photos together: A resembles B, B resembles C, and so on. The largest group, 77 plastic photos, is such a chain, and very likely holds several different objects. StratifiedGroupKFold had to put it somewhere, and it fell into test, where it is the whole plastic class. Two consequences:

- the plastic test score describes that family of look-alike photos more than plastic in general;
- confidence intervals that resample whole groups treat those 77 photos as one observation, which is too cautious. Intervals that resample single photos are too optimistic. The README reports both.

## 4. Version 1 of the split

The first split used average-linkage clusters only. A review found 44 test photos with a same-class training photo at ≥ 0.86 (11 at ≥ 0.88), some visibly the same object. That model scored 100% on those 44 photos and 95.0% on the other 317. The split was rebuilt as described above and the models retrained. The version 1 split, results and model record are in [history/v1](history/v1).
