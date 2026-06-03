"""Computes FLOPs and parameter counts for all models."""
import torch
from thop import profile
from models.mixer import MLPMixer
from models.efficientnet import get_efficientnet, get_resnet
from models.mixer_pretrained import get_pretrained_mixer


def count_flops(model: torch.nn.Module, input_size: tuple = (1, 3, 224, 224)) -> tuple:
    """Counts FLOPs and parameters for a given model.

    Args:
        model: The neural network model to profile.
        input_size: Input tensor shape as (batch, channels, height, width).

    Returns:
        Tuple of (flops, params) as raw counts.
    """
    dummy_input = torch.randn(*input_size)
    model.eval()
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    return flops, params


def main() -> None:
    """Profiles all models and prints a comparison table."""
    models_dict = {
        "MLP-Mixer (scratch)":     MLPMixer(image_size=224, patch_size=16,
                                            num_classes=10, hidden_dim=512,
                                            num_layers=8, tokens_mlp_dim=256,
                                            channels_mlp_dim=2048),
        "EfficientNet-B0":         get_efficientnet(num_classes=10),
        "ResNet-50":               get_resnet(num_classes=10),
        "Mixer-B/16 (pretrained)": get_pretrained_mixer(num_classes=10),
    }

    print(f"\n{'Model':<30} {'GFLOPs':>10} {'Params (M)':>12}")
    print("-" * 55)
    for name, model in models_dict.items():
        flops, params = count_flops(model)
        print(f"{name:<30} {flops/1e9:>10.2f} {params/1e6:>12.2f}")
    print()


if __name__ == "__main__":
    main()