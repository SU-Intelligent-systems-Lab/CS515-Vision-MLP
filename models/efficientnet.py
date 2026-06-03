import torch.nn as nn
import torchvision.models as models


def get_efficientnet(
    num_classes: int = 10,
    pretrained: bool = False
) -> nn.Module:
    """Builds an EfficientNet-B0 model for image classification.

    Loads EfficientNet-B0 from torchvision and replaces the
    classifier head to match the desired number of output classes.

    Args:
        num_classes: Number of output classes for the classifier head.
        pretrained: Whether to load ImageNet pretrained weights.

    Returns:
        EfficientNet-B0 model with a custom classifier head.
    """
    model = models.efficientnet_b0(
        weights="IMAGENET1K_V1" if pretrained else None
    )
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, num_classes
    )
    return model

def get_resnet(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    """Builds a ResNet-50 model for image classification.

    Loads ResNet-50 from torchvision and replaces the fully connected
    head to match the desired number of output classes.

    Args:
        num_classes: Number of output classes for the classifier head.
        pretrained: Whether to load ImageNet pretrained weights.

    Returns:
        ResNet-50 model with a custom classifier head.
    """
    model = models.resnet50(
        weights="IMAGENET1K_V1" if pretrained else None
    )
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model