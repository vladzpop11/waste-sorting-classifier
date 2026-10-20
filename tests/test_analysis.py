import numpy as np
import pytest


def test_selection_rule() -> None:
    from waste_sorting_classifier.select import decide

    assert (
        decide({"resnet18": 0.93, "efficientnet_b0": 0.95}, 0.01, "efficientnet_b0")
        == "efficientnet_b0"
    )
    assert (
        decide({"resnet18": 0.95, "efficientnet_b0": 0.93}, 0.01, "efficientnet_b0") == "resnet18"
    )
    assert (
        decide({"resnet18": 0.945, "efficientnet_b0": 0.94}, 0.01, "efficientnet_b0")
        == "efficientnet_b0"
    )


def test_reliability_measures_the_confidence_gap() -> None:
    from waste_sorting_classifier.metrics import reliability

    y = np.array([0, 0, 1, 1])
    sure_and_right = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    assert reliability(sure_and_right, y)["ece"] == 0
    # 90% confident on every photo but right on half: a gap of 0.4.
    overconfident = np.array([[0.9, 0.1], [0.1, 0.9], [0.1, 0.9], [0.9, 0.1]])
    table = reliability(overconfident, y)
    assert abs(table["ece"] - 0.4) < 1e-9
    assert table["bins"] == [{"from": 0.8, "to": 0.9, "photos": 4, "accuracy": 0.5,
                              "confidence": 0.9}]  # fmt: skip


def test_low_confidence_threshold_is_the_lowest_meeting_the_target() -> None:
    from waste_sorting_classifier.metrics import low_confidence_threshold

    confidence = np.array([0.52, 0.58, 0.66, 0.71, 0.9, 0.95, 0.97, 0.99])
    correct = np.array([False, True, False, True, True, True, True, True])
    flag = low_confidence_threshold(confidence, correct, target=0.99)
    assert flag["threshold"] == 0.7  # 0.65 still includes the 0.66 error
    assert flag["accuracy_above"] == 1.0
    assert flag["accuracy_below"] == 1 / 3
    assert flag["share_below"] == 3 / 8


def test_grouped_intervals_treat_a_group_as_one_observation() -> None:
    from waste_sorting_classifier.metrics import bootstrap_intervals

    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 60)
    pred = np.where(rng.random(60) < 0.8, y, (y + 1) % 3)
    single = bootstrap_intervals(y, pred, np.arange(60), 3, seed=1)["per_group"]
    # Five identical copies of every photo, grouped: no new evidence, so the same intervals.
    copies = bootstrap_intervals(np.repeat(y, 5), np.repeat(pred, 5),
                                 np.repeat(np.arange(60), 5), 3, seed=1)  # fmt: skip
    assert copies["per_group"]["accuracy"] == single["accuracy"]
    assert copies["per_group"]["per_class_f1"] == single["per_class_f1"]
    # Treated as independent photos, the copies give a narrower, overconfident interval.
    width = lambda ci: ci[1] - ci[0]  # noqa: E731
    assert width(copies["per_photo"]["accuracy"]) < 0.6 * width(single["accuracy"])
    low, high = single["accuracy"]
    assert low <= (y == pred).mean() <= high


def test_a_class_missing_from_a_resample_is_not_scored_zero() -> None:
    from waste_sorting_classifier.metrics import bootstrap_intervals

    # Perfect predictions; class 2 is a single group, so many resamples leave it out.
    y = np.array([0] * 20 + [1] * 20 + [2] * 10)
    groups = np.concatenate([np.arange(40), np.full(10, 99)])
    ci = bootstrap_intervals(y, y.copy(), groups, 3, seed=0)["per_group"]
    assert ci["macro_f1"] == [1.0, 1.0]
    assert ci["per_class_f1"][2] == [1.0, 1.0]


def test_low_confidence_threshold_is_none_when_no_threshold_meets_the_target() -> None:
    from waste_sorting_classifier.metrics import low_confidence_threshold

    flag = low_confidence_threshold(np.array([0.6, 0.7, 0.96, 0.97, 0.98]),
                                    np.array([True, True, False, True, True]))  # fmt: skip
    assert flag["threshold"] is None


def test_version_4_score_counts_each_source_equally() -> None:
    from waste_sorting_classifier.select import score

    metrics = {"val_macro_f1": 0.90, "val_macro_f1_trashnet": 0.96, "val_macro_f1_realwaste": 0.80}
    rule = {"score": "mean_over_sources", "sources": ["trashnet", "realwaste"]}
    assert score(metrics, rule) == pytest.approx(0.88)
    assert score(metrics, {}) == 0.90  # version 2's rule: plain macro F1
