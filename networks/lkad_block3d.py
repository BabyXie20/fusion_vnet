"""
3D implementation of the LKA_d attention block.

The block preserves spatial dimensions while applying a series of depth-wise
convolutions to capture large receptive fields, followed by channel mixing
via point-wise convolutions. It is tailored for 3D medical image segmentation
tasks where volumetric consistency is required.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DWConvBlock3D(nn.Module):
    """
    Depth-wise convolution block for 3D tensors.

    This corresponds to the "DW-Conv" component (3x3x3 depth-wise convolution)
    in the original LKA_d design.
    Input / Output: (B, C, D, H, W), same spatial size.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.norm = nn.InstanceNorm3d(channels, affine=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        return self.act(x)


class DWDConv3D(nn.Module):
    """
    Depth-wise dilated 3D convolution.

    This implements the "DWD-Conv" stage with dilation to enlarge the
    receptive field while preserving spatial dimensions.
    """

    def __init__(self, channels: int, dilation: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv3d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.norm = nn.InstanceNorm3d(channels, affine=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        return self.act(x)


class DDWConv3D(nn.Module):
    """
    Approximate deformable depth-wise 3D convolution.

    This block approximates a deformable depth-wise convolution ("DDW-Conv3")
    using two sequential depth-wise convolutions with larger kernels to mimic
    a wider effective receptive field.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv3d(
                channels,
                channels,
                kernel_size=5,
                padding=2,
                groups=channels,
                bias=False,
            ),
            nn.InstanceNorm3d(channels, affine=True),
            nn.ReLU(inplace=True),
        )
        self.stage2 = nn.Sequential(
            nn.Conv3d(
                channels,
                channels,
                kernel_size=7,
                padding=3,
                groups=channels,
                bias=False,
            ),
            nn.InstanceNorm3d(channels, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        return x


class LKAdBlock3D(nn.Module):
    """
    3D LKA_d block for medical image segmentation.

    Input:  (B, C, D, H, W)
    Output: (B, C, D, H, W)  with residual connection.
    """

    def __init__(self, in_channels: int, out_channels: int | None = None) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels

        # Conv1: point-wise convolution to mix channels without changing spatial dims.
        self.conv_in = nn.Conv3d(self.in_channels, self.out_channels, kernel_size=1, bias=False)
        self.act = nn.ReLU(inplace=True)

        # Local attention branch: DW-Conv -> DWD-Conv -> DDW-Conv3 -> Conv1.
        self.dw_conv = DWConvBlock3D(self.out_channels)
        self.dwd_conv = DWDConv3D(self.out_channels)
        self.ddw_conv = DDWConv3D(self.out_channels)
        self.conv_attn = nn.Conv3d(self.out_channels, self.out_channels, kernel_size=1, bias=False)

        # Output projection after attention gating.
        self.conv_out = nn.Conv3d(self.out_channels, self.out_channels, kernel_size=1, bias=False)

        # Optional residual projection when channel dimensions differ.
        self.residual_proj = None
        if self.in_channels != self.out_channels:
            self.residual_proj = nn.Conv3d(self.in_channels, self.out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        x = self.conv_in(x)
        g = self.act(x)

        # Local attention branch
        u = self.dw_conv(g)
        u = self.dwd_conv(u)
        u = self.ddw_conv(u)
        u = self.conv_attn(u)

        # Attention gating
        v = u * g

        # Output projection
        y = self.conv_out(v)

        # Residual connection with optional projection
        if self.residual_proj is not None:
            identity = self.residual_proj(identity)

        out = y + identity
        return out


if __name__ == "__main__":
    # Simple self-test to verify spatial dimensions are preserved.
    dummy = torch.randn(2, 64, 32, 128, 128)
    block = LKAdBlock3D(64)
    out = block(dummy)
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {out.shape}")
