import numpy as np


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


def test_grouped_bootstrap_treats_a_group_as_one_observation() -> None:
    from waste_sorting_classifier.metrics import grouped_bootstrap

    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 60)
    pred = np.where(rng.random(60) < 0.8, y, (y + 1) % 3)
    single = grouped_bootstrap(y, pred, np.arange(60), 3, seed=1)
    # Five identical copies of every photo, grouped: no new evidence, so the same intervals.
    copies = grouped_bootstrap(np.repeat(y, 5), np.repeat(pred, 5), np.repeat(np.arange(60), 5),
                               3, seed=1)  # fmt: skip
    assert copies["accuracy"] == single["accuracy"]
    assert copies["per_class_f1"] == single["per_class_f1"]
    # Treated as independent photos, the copies would give a narrower, overconfident interval.
    naive = grouped_bootstrap(np.repeat(y, 5), np.repeat(pred, 5), np.arange(300), 3, seed=1)
    width = lambda ci: ci[1] - ci[0]  # noqa: E731
    assert width(naive["accuracy"]) < 0.6 * width(single["accuracy"])
    low, high = single["accuracy"]
    assert low <= (y == pred).mean() <= high
