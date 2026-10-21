# Data: duplicates, repeated objects and the split

Sections 1 to 4 describe TrashNet, whose split has not changed since version 2. Section 5 adds RealWaste (version 4).

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

## 5. Version 4: RealWaste

**The dataset.** [RealWaste](https://archive.ics.uci.edu/dataset/908/realwaste) (Single, Iranmanesh & Raad, 2023): 4,752 photos of waste received at the Whyte's Gully landfill in Wollongong, Australia, each item photographed from above on a concrete floor, 524 × 524 pixels. `make data` checks the archive's SHA-256 and the published counts per class. All files are distinct (no byte-identical duplicates) and none carries an EXIF orientation.

**Licence.** UCI lists CC BY 4.0, but the authors' own README says CC BY-NC-SA 4.0. The project follows the stricter one: non-commercial use, attribution, and the same licence for adaptations, which is why the version 4 model weights are CC BY-NC-SA 4.0.

**Classes.** RealWaste's cardboard, glass, metal, paper and plastic map to ours. Its miscellaneous trash (495) and textile trash (318) become trash. Its food organics (411) and vegetation (436) have no counterpart and are not used. That leaves 3,905 photos, and trash grows from 137 photos (TrashNet) to 950.

**Repeated objects.** Like TrashNet, RealWaste photographs many objects more than once. The same threshold, 0.86, is used, based on RealWaste's own pairs ([figures/similarity_pairs_realwaste.png](figures/similarity_pairs_realwaste.png), counts in [data_stats.json](data_stats.json)):

| Cosine similarity | Same-class pairs | What the pairs are |
| --- | --- | --- |
| ≥ 0.95 | 18 | The same object, barely moved |
| 0.90–0.95 | 662 | The same object, re-posed |
| 0.88–0.90 | 1,242 | Almost always the same object |
| 0.86–0.88 | 2,688 | Often the same object, plus look-alikes such as crumpled foil |
| 0.84–0.86 | 5,182 | Mostly different objects of the same kind (different cans, different foil) |
| 0.80–0.84 | 24,516 | Different objects |

**Placing RealWaste photos without disturbing TrashNet.** TrashNet keeps its version 2 split exactly ([splits/trashnet-v2.csv](../splits/trashnet-v2.csv)). RealWaste photos are grouped among themselves, then:
- **Near-duplicates across sources:** 13 RealWaste photos have a TrashNet photo at ≥ 0.86 in the same class, most likely the same product (for example the same can). Their groups follow that TrashNet photo's split. One group was tied both to a TrashNet training photo and to a TrashNet validation photo, through different members; its member tied to validation (`realwaste/metal/Metal_582.jpg`) is left out and the group goes to training.
- **Look-alike chains go to training.** On RealWaste's concrete backgrounds, joining every pair at ≥ 0.86 chains look-alikes into large groups: the largest has 263 photos, and 10 groups have more than 20 (1,102 photos in all). On a first attempt StratifiedGroupKFold put a 130-photo glass chain into the test fold, where it was the entire glass test set, and a 56-photo metal chain into validation. The rule, fixed before any training or evaluation and based on the split's structure only: groups of more than 20 photos go to training. Validation and test then consist of many independent objects.
- **The rest** is split with StratifiedGroupKFold (7 folds, as for TrashNet): 3,109 training, 398 validation and 397 test photos, with each class at 8–14% of its photos in each of validation and test.

**The guarantee, over both sources.** No validation or test photo has a same-class training photo, from either source, at ≥ 0.86. The highest is 0.8599 for validation and 0.8596 for test ([split_stats.json](split_stats.json)). Just below the threshold:

| Nearest same-class training photo | TrashNet val (360) | TrashNet test (366) | RealWaste val (398) | RealWaste test (397) |
| --- | --- | --- | --- | --- |
| ≥ 0.85 | 32 | 47 | 74 | 60 |
| ≥ 0.84 | 76 | 102 | 144 | 128 |
| ≥ 0.80 | 245 | 272 | 334 | 335 |

