import torch
import torch.nn as nn

class TorchResBlock3D(nn.Module):
    """A lightweight 3D residual block implemented purely with torch layers."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(channels),
        )
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + residual
        return self.act(out)


class TransformerBlock_3D_LKA(nn.Module):
    """
    A transformer block, based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        proj_size: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        pos_embed=False,
    ) -> None:
        """
        Args:
            input_size: the size of the input for each stage.
            hidden_size: dimension of hidden layer.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            pos_embed: bool argument to determine if positional embedding is used.

        """

        super().__init__()
        print("Using LKA Attention with different Kernel sizes.")

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            print("Hidden size is ", hidden_size)
            print("Num heads is ", num_heads)
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.norm = nn.LayerNorm(hidden_size)
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)
        self.epa_block = LKA_Attention3d(d_model=hidden_size)
        self.conv51 = TorchResBlock3D(hidden_size)
        self.conv8 = nn.Sequential(
            nn.Dropout3d(0.1, False), nn.Conv3d(hidden_size, hidden_size, 1)
        )

        self.pos_embed = None
        if pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size))

    def forward(self, x):
        B, C, H, W, D = x.shape

        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        attn = x + self.gamma * self.epa_block(self.norm(x), B, C, H, W, D)

        attn_skip = attn.reshape(B, H, W, D, C).permute(
            0, 4, 1, 2, 3
        )  # (B, C, H, W, D)
        attn = self.conv51(attn_skip)
        x = attn_skip + self.conv8(attn)

        return x
    
class LKA3d(nn.Module):
    def __init__(self, dim):
        super().__init__()

        kernel_dwd = 7
        dilation_dwd = 3
        padding_dwd = 9
        kernel_dw = 5
        padding_dw = 2
        """
        if dim == 32 or dim == 64:
            kernel_dwd = 7
            dilation_dwd = 3
            padding_dwd = 9
            kernel_dw = 5
            padding_dw = 2
        elif dim == 128:
            kernel_dwd = 5
            dilation_dwd = 3
            padding_dwd = 6
            kernel_dw = 5
            padding_dw = 2
        elif dim == 256:
            kernel_dwd = 3
            dilation_dwd = 2
            padding_dwd = 2
            kernel_dw = 3
            padding_dw = 1
        else:
            raise ValueError("Unknown dim: {}".format(dim))
        """

        self.conv0 = nn.Conv3d(
            dim, dim, kernel_size=kernel_dw, padding=padding_dw, groups=dim
        )
        self.conv_spatial = nn.Conv3d(
            dim,
            dim,
            kernel_size=kernel_dwd,
            stride=1,
            padding=padding_dwd,
            groups=dim,
            dilation=dilation_dwd,
        )
        self.conv1 = nn.Conv3d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn


class LKA_Attention3d(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv3d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA3d(d_model)
        self.proj_2 = nn.Conv3d(d_model, d_model, 1)

    def forward(self, x, B, C, H, W, D):
        x = x.permute(0, 2, 1).reshape(B, C, H, W, D)  # B N C --> B C N --> B C H W D
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shortcut
        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)
        return x
