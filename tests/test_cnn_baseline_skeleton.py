"""Serving builds the architecture only; weights come from the checkpoint.

Before this, constructing the served backbone read the gitignored Places365
.tar via a CWD-relative path and fell back to downloading ImageNet weights -
both immediately overwritten by the checkpoint, both fatal in a container.
"""
import pytest
import torch
import torchvision.models._api as tv_api

from models.cnn_baseline import CNNBaseline


def _forbid(*_args, **_kwargs):
    raise AssertionError("pretrained weights must not be read or downloaded")


@pytest.fixture
def no_weights(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # models/resnet50_places365_weights.pth.tar unreachable
    monkeypatch.setattr(tv_api, "load_state_dict_from_url", _forbid)
    monkeypatch.setattr(torch.hub, "load", _forbid)


def test_skeleton_builds_without_weights(no_weights):
    model = CNNBaseline(67, backbone="resnet50_places365_local", pretrained=False)
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 224, 224))
    assert out.shape == (1, 67)


def test_default_still_loads_pretrained(no_weights):
    with pytest.raises(AssertionError, match="must not be read"):
        CNNBaseline(67, backbone="resnet50_places365_local")


def test_skeleton_rejects_unknown_backbone():
    with pytest.raises(ValueError):
        CNNBaseline(67, backbone="vgg16", pretrained=False)
