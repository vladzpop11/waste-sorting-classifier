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
