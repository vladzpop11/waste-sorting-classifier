import pytest

from waste_sorting_classifier.artifacts import sha256
from waste_sorting_classifier.fetch_model import fetch


@pytest.fixture
def published(tmp_path):
    source = tmp_path / "release" / "model.onnx"
    source.parent.mkdir()
    source.write_bytes(b"the evaluated model")
    return source


def test_fetch_installs_a_matching_file(published, tmp_path) -> None:
    dest = tmp_path / "models" / "model.onnx"
    fetch(published.as_uri(), dest, sha256(published))
    assert dest.read_bytes() == b"the evaluated model"
    assert list(dest.parent.iterdir()) == [dest]


def test_fetch_refuses_a_different_file_and_keeps_the_old_one(published, tmp_path) -> None:
    dest = tmp_path / "models" / "model.onnx"
    dest.parent.mkdir()
    dest.write_bytes(b"local export")
    with pytest.raises(ValueError, match="expected"):
        fetch(published.as_uri(), dest, "0" * 64)
    assert dest.read_bytes() == b"local export"
    assert list(dest.parent.iterdir()) == [dest]
