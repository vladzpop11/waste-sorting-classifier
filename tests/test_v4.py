"""Version 4 logic, on small constructed cases: placing RealWaste photos around the frozen
TrashNet split, the evaluation's paired comparison and summaries, and the Mahalanobis
familiarity score."""

import numpy as np
import pandas as pd
import pytest

from waste_sorting_classifier.split import max_cross_split_similarity, place_realwaste

DIMS = 64
CFG = {"split": {"similarity_threshold": 0.86, "chains_to_train_above": 20, "n_folds": 7},
       "seed": 0}  # fmt: skip


def axis(k: int) -> np.ndarray:
    v = np.zeros(DIMS)
    v[k] = 1.0
    return v


def in_plane(degrees: float) -> np.ndarray:
    """A unit vector in the plane of axes 0 and 1, at the given angle from axis 0."""
    r = np.radians(degrees)
    return np.cos(r) * axis(0) + np.sin(r) * axis(1)


@pytest.fixture
def case():
    """TrashNet (frozen): t-train at 0 deg and t-val at 60 deg in one plane (cosine 0.5, so
    not near-duplicates), another validation photo on its own axis, a test photo of class b.
    RealWaste:
    - near-val: a near-duplicate of the second validation photo -> must follow it to val;
    - a chain at 15, 30 and 45 deg (one group): 15 deg is tied to t-train, 45 deg to t-val,
      30 deg to both -> the group goes to train and the photos tied to val are left out;
    - 22 near-identical class-b photos (one group of more than 20) -> train;
    - 14 independent photos per class -> spread by StratifiedGroupKFold."""
    rows, vectors = [], []

    def add(path, label, source, v):
        rows.append({"path": path, "label": label, "source": source})
        vectors.append(v / np.linalg.norm(v))

    add("t-train", "a", "trashnet", in_plane(0))
    add("t-val", "a", "trashnet", in_plane(60))
    add("t-val2", "a", "trashnet", axis(8))
    add("t-test", "b", "trashnet", axis(9))
    near = np.cos(np.radians(10)) * axis(8) + np.sin(np.radians(10)) * axis(40)  # cosine 0.98
    add("r-near-val", "a", "realwaste", near)
    for deg in (15, 30, 45):
        add(f"r-chain-{deg}", "a", "realwaste", in_plane(deg))
    rng = np.random.default_rng(0)
    for i in range(22):
        add(f"r-big-{i}", "b", "realwaste", axis(5) + 0.05 * rng.normal(size=DIMS))
    for i in range(14):
        add(f"r-free-a{i}", "a", "realwaste", axis(10 + i))
        add(f"r-free-b{i}", "b", "realwaste", axis(30 + i % 10) + axis(50 + i % 14))
    index = pd.DataFrame(rows)
    frozen = pd.DataFrame({"path": ["t-train", "t-val", "t-val2", "t-test"],
                           "group": ["a-0", "a-1", "a-2", "b-0"],
                           "split": ["train", "val", "val", "test"]})  # fmt: skip
    return index, np.array(vectors), frozen


def test_trashnet_keeps_its_frozen_split(case) -> None:
    index, emb, frozen = case
    placed, _ = place_realwaste(index, emb, frozen, CFG)
    trashnet = placed[placed["source"] == "trashnet"].set_index("path")
    assert trashnet["split"].to_dict() == frozen.set_index("path")["split"].to_dict()


def test_a_near_duplicate_of_a_trashnet_photo_follows_its_split(case) -> None:
    index, emb, frozen = case
    placed, _ = place_realwaste(index, emb, frozen, CFG)
    assert placed.set_index("path").loc["r-near-val", "split"] == "val"


def test_a_group_tied_to_two_splits_goes_to_train_without_its_val_ties(case) -> None:
    index, emb, frozen = case
    placed, left_out = place_realwaste(index, emb, frozen, CFG)
    assert sorted(left_out) == ["r-chain-30", "r-chain-45"]  # tied to t-val (30: to both)
    assert placed.set_index("path").loc["r-chain-15", "split"] == "train"


def test_look_alike_chains_go_to_train(case) -> None:
    index, emb, frozen = case
    placed, _ = place_realwaste(index, emb, frozen, CFG)
    big = placed[placed["path"].str.startswith("r-big-")]
    assert big["group"].nunique() == 1 and (big["split"] == "train").all()


def test_independent_photos_are_spread_and_the_guarantee_holds(case) -> None:
    index, emb, frozen = case
    placed, _ = place_realwaste(index, emb, frozen, CFG)
    free = placed[placed["path"].str.startswith("r-free-")]
    assert set(free["split"]) == {"train", "val", "test"}
    worst = max_cross_split_similarity(
        emb[placed.index.to_numpy()], placed["label"].to_numpy(), placed["split"].to_numpy()
    )
    assert max(worst.values()) < CFG["split"]["similarity_threshold"]


def test_paired_difference_is_zero_for_the_same_answers_and_exact_otherwise() -> None:
    from waste_sorting_classifier.evaluate import paired_difference

    rng = np.random.default_rng(1)
    a = rng.random(200) < 0.9
    groups = np.repeat(np.arange(50), 4)
    same = paired_difference(a, a, groups, seed=0)
    assert same["difference"] == 0 and same["ci95"] == [0.0, 0.0]
    b = a.copy()
    b[:20] = False  # b is wrong on the first 20 photos
    d = paired_difference(a, b, groups, seed=0)
    assert d["difference"] == pytest.approx(a[:20].mean() * 20 / 200)
    assert d["ci95"][0] <= d["difference"] <= d["ci95"][1]


def test_summary_scores_match_direct_counts() -> None:
    from waste_sorting_classifier.evaluate import summary

    classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
    rng = np.random.default_rng(2)
    n = 120
    y = rng.integers(0, 6, n)
    pred = np.where(rng.random(n) < 0.8, y, (y + 1) % 6)
    proba = np.full((n, 6), 0.02)
    proba[np.arange(n), pred] = 0.9
    rows = pd.DataFrame({"label": [classes[k] for k in y], "group": np.arange(n) // 2,
                         "similarity": rng.uniform(0.70, 0.859, n)})  # fmt: skip
    cfg = {"split": {"similarity_threshold": 0.86}, "seed": 0}
    s = summary(rows, proba, classes, cfg)
    assert s["accuracy"] == pytest.approx((pred == y).mean())
    assert sum(b["photos"] for b in s["accuracy_by_similarity_to_train"]) == n
    assert s["per_class"]["glass"]["support"] == int((y == 1).sum())
    low, high = s["ci95"]["per_group"]["accuracy"]
    assert low <= s["accuracy"] <= high


def test_mahalanobis_matches_a_direct_computation(tmp_path) -> None:
    from waste_sorting_classifier.artifacts import sha256
    from waste_sorting_classifier.familiarity import Familiarity, mahalanobis

    rng = np.random.default_rng(3)
    means = rng.normal(size=(6, 5))
    a = rng.normal(size=(5, 5))
    precision = a @ a.T + np.eye(5)  # symmetric positive definite
    features = rng.normal(size=(4, 5))
    direct = -np.array([min((f - m) @ precision @ (f - m) for m in means) for f in features])
    assert mahalanobis(means, precision, features) == pytest.approx(direct)

    # Stored as float32 and read back by the API's loader.
    stored = np.concatenate([means.ravel(), precision.ravel()]).astype("<f4")
    stored.tofile(tmp_path / "familiarity.bin")
    meta = {"familiarity": {"method": "mahalanobis", "file": "familiarity.bin",
                            "sha256": sha256(tmp_path / "familiarity.bin"), "classes": 6,
                            "dims": 5, "threshold": -10.0}}  # fmt: skip
    fam = Familiarity(tmp_path, meta)
    assert fam.score(None, features) == pytest.approx(direct, rel=1e-5)
    meta["familiarity"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        Familiarity(tmp_path, meta)
