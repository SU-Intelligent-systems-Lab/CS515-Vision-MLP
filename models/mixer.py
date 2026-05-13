import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class MlpBlock(nn.Module):
    """A two-layer MLP block with GELU activation.

    Used as the basic building block for both token-mixing
    and channel-mixing operations in the Mixer architecture.

    Attributes:
        fc1: First fully connected layer projecting to hidden_dim.
        act: GELU activation function.
        fc2: Second fully connected layer projecting back to input_dim.
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        """Initializes MlpBlock with two linear layers and GELU.

        Args:
            input_dim: Dimensionality of input and output.
            hidden_dim: Dimensionality of the hidden layer.
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP block.

        Args:
            x: Input tensor of shape (batch, *, input_dim).

        Returns:
            Output tensor of same shape as input.
        """
        return self.fc2(self.act(self.fc1(x)))


class MixerBlock(nn.Module):
    """One full Mixer layer with token-mixing and channel-mixing.

    Applies token-mixing MLP across spatial locations followed by
    channel-mixing MLP across feature dimensions, each with a
    skip connection and layer normalization.

    Attributes:
        norm1: Layer normalization before token-mixing.
        norm2: Layer normalization before channel-mixing.
        token_mixing: MLP applied across the patch dimension.
        channel_mixing: MLP applied across the feature dimension.
    """

    def __init__(
        self,
        num_patches: int,
        hidden_dim: int,
        tokens_mlp_dim: int,
        channels_mlp_dim: int
    ) -> None:
        """Initializes MixerBlock with token and channel mixing MLPs.

        Args:
            num_patches: Number of image patches (sequence length).
            hidden_dim: Feature dimensionality of each patch.
            tokens_mlp_dim: Hidden dimension of the token-mixing MLP.
            channels_mlp_dim: Hidden dimension of the channel-mixing MLP.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.token_mixing = MlpBlock(num_patches, tokens_mlp_dim)
        self.channel_mixing = MlpBlock(hidden_dim, channels_mlp_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through one Mixer layer.

        Args:
            x: Input tensor of shape (batch, num_patches, hidden_dim).

        Returns:
            Output tensor of same shape as input.
        """
        y = self.norm1(x)
        y = y.transpose(1, 2)
        y = self.token_mixing(y)
        y = y.transpose(1, 2)
        x = x + y

        y = self.norm2(x)
        y = self.channel_mixing(y)
        x = x + y
        return x


class MLPMixer(nn.Module):
    """MLP-Mixer architecture for image classification.

    An all-MLP architecture that uses alternating token-mixing and
    channel-mixing MLPs to process image patches without convolutions
    or self-attention. Based on Tolstikhin et al. (2021).

    Attributes:
        patch_embed: Patch embedding layer.
        mixer_layers: Sequence of MixerBlock layers.
        norm: Final layer normalization.
        head: Linear classification head.
        dropout: Dropout layer applied after patch embedding.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        num_classes: int = 1000,
        hidden_dim: int = 512,
        num_layers: int = 8,
        tokens_mlp_dim: int = 256,
        channels_mlp_dim: int = 2048,
        dropout: float = 0.0,
    ) -> None:
        """Initializes MLPMixer with specified architecture parameters.

        Args:
            image_size: Input image resolution (assumes square images).
            patch_size: Size of each square image patch.
            num_classes: Number of output classes.
            hidden_dim: Feature dimensionality for all patch embeddings.
            num_layers: Number of Mixer layers to stack.
            tokens_mlp_dim: Hidden dimension of token-mixing MLPs.
            channels_mlp_dim: Hidden dimension of channel-mixing MLPs.
            dropout: Dropout probability applied after patch embedding.
        """
        super().__init__()
        assert image_size % patch_size == 0, \
            "Image size must be divisible by patch size"
        num_patches = (image_size // patch_size) ** 2

        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, hidden_dim,
                      kernel_size=patch_size, stride=patch_size),
            Rearrange("b c h w -> b (h w) c"),
        )
        self.mixer_layers = nn.Sequential(*[
            MixerBlock(num_patches, hidden_dim,
                       tokens_mlp_dim, channels_mlp_dim)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the full MLP-Mixer model.

        Args:
            x: Input image tensor of shape (batch, 3, image_size, image_size).

        Returns:
            Class logits tensor of shape (batch, num_classes).
        """
        x = self.patch_embed(x)
        x = self.dropout(x)
        x = self.mixer_layers(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)