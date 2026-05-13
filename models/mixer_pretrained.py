import torch.nn as nn
import timm


def get_pretrained_mixer(num_classes: int = 10) -> nn.Module:
    """Builds a pretrained MLP-Mixer-B/16 model for image classification.

    Loads Mixer-B/16 pretrained on ImageNet-21k via the timm library
    and replaces the classification head to match the desired number
    of output classes.

    Args:
        num_classes: Number of output classes for the classifier head.

    Returns:
        Pretrained Mixer-B/16 model with a custom classifier head.
    """
    model = timm.create_model(
        "mixer_b16_224_miil_in21k",
        pretrained=True
    )
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model