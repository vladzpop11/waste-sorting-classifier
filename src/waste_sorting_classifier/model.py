"""Pre-trained torchvision backbones with a new classification head."""

import torch
import torchvision

ARCHS = {
    "resnet18": (torchvision.models.resnet18, torchvision.models.ResNet18_Weights.IMAGENET1K_V1),
    "efficientnet_b0": (
        torchvision.models.efficientnet_b0,
        torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1,
    ),
}


def build_model(arch: str, num_classes: int, pretrained: bool = True) -> torch.nn.Module:
    if arch not in ARCHS:
        raise ValueError(f"unknown arch {arch!r}; choose from {sorted(ARCHS)}")
    factory, weights = ARCHS[arch]
    model = factory(weights=weights if pretrained else None)
    if arch == "resnet18":
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    else:
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
    return model
