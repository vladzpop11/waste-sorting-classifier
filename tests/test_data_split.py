import numpy as np
import pandas as pd
import pytest

from waste_sorting_classifier.data import check_counts, drop_conflicting_duplicates
from waste_sorting_classifier.split import assign_splits, group_similar


def test_conflicting_duplicates_are_dropped_but_same_label_copies_are_not() -> None:
    index = pd.DataFrame(
        {
            "path": ["glass/a", "metal/a", "paper/b", "paper/c", "paper/d"],
            "label": ["glass", "metal", "paper", "paper", "paper"],
            "sha256": ["x", "x", "y", "y", "z"],
        }
    )
    kept, dropped = drop_conflicting_duplicates(index)
    assert sorted(dropped["path"]) == ["glass/a", "metal/a"]
    assert sorted(kept["path"]) == ["paper/b", "paper/c", "paper/d"]


def test_counts_must_match_the_published_ones() -> None:
    index = pd.DataFrame({"label": ["a", "a", "b"]})
    check_counts(index, {"a": 2, "b": 1})
    with pytest.raises(ValueError, match="published counts"):
        check_counts(index, {"a": 2, "b": 2})


def test_grouping_joins_near_copies_within_a_class_only() -> None:
    rng = np.random.default_rng(0)
    base = rng.normal(size=(4, 16))
    emb = np.vstack([base[0], base[0] + 0.01, base[1], base[2], base[2] + 0.01])
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    labels = np.array(["a", "a", "a", "b", "a"])  # rows 3 and 4 are near copies, other class
    groups = group_similar(emb, labels, threshold=0.99)
    assert groups[0] == groups[1]
    assert groups[0] != groups[2]
    assert groups[3] != groups[4]


def test_splits_keep_groups_together_and_stay_stratified() -> None:
    labels = np.repeat(["a", "b", "c"], [300, 200, 100])
    groups = np.array([f"{lab}-{i // 3}" for i, lab in enumerate(labels)])  # groups of 3
    split = assign_splits(labels, groups, n_folds=7, seed=42)
    df = pd.DataFrame({"label": labels, "group": groups, "split": split})
    assert (df.groupby("group")["split"].nunique() == 1).all()
    shares = df.groupby("split").size() / len(df)
    assert shares["test"] == pytest.approx(1 / 7, abs=0.03)
    assert shares["val"] == pytest.approx(1 / 7, abs=0.03)
    for label, rate in (pd.crosstab(df["label"], df["split"], normalize="index")["test"]).items():
        assert rate == pytest.approx(1 / 7, abs=0.04), label
