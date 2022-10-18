"""A simple U-Net with timm backbone encoder.

Based off an old version of Unet in
    https://github.com/qubvel/segmentation_models.pytorch
Hacked together by Ross Wightman
    https://gist.github.com/rwightman/f8b24f4e6f5504aba03e999e02460d31
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Unet(nn.Module):
    """Unet is a fully convolutional neural network for image semantic segmentation.

    NOTE: This is based off an old version of Unet in
    https://github.com/qubvel/segmentation_models.pytorch

    Parameters
    ----------
    encoder
        Classification model (without last dense layers) used as feature extractor to build segmentation model.
    num_classes
        Number of classes for output (output shape - `(batch, classes, h, w)`).
    decoder_channels
        List of numbers of ``Conv2D`` layer filters in decoder blocks
    decoder_use_batchnorm
        If True, use `BatchNormalisation` layer between `Conv2D` and `Activation` layers.
    center
        If True, add `Conv2dReLU` block on encoder head.
    """

    def __init__(
        self,
        encoder,
        num_classes,
        decoder_use_batchnorm=True,
        decoder_channels=(256, 128, 64, 32, 16),
        center=False,
        norm_layer=nn.BatchNorm2d,
        scale_factors=None,
    ):
        super().__init__()
        # backbone_kwargs = backbone_kwargs or {}
        # # NOTE: some models need different backbone indices specified based on
        # # the alignment of features and some models won't have a full enough range
        # # of feature strides to work properly.
        # encoder = timm.create_model(
        #     backbone,
        #     features_only=True,
        #     out_indices=backbone_indices,
        #     in_chans=in_chans,
        #     pretrained=True,
        #     **backbone_kwargs
        # )
        encoder_channels = encoder.feature_info.channels()[::-1]
        self.encoder = encoder
        self.default_cfg = self.encoder.default_cfg

        if not decoder_use_batchnorm:
            norm_layer = None
        self.decoder = UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            final_channels=num_classes,
            norm_layer=norm_layer,
            center=center,
            scale_factors=scale_factors,
        )

    def forward(self, x: torch.Tensor):
        """Run forward pass."""
        x = self.encoder(x)
        x.reverse()  # torchscript doesn't work with [::-1]
        x = self.decoder(x)
        return x


class Conv2dBnAct(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        act_layer=nn.ReLU,
        norm_layer=nn.BatchNorm2d,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = norm_layer(out_channels)
        self.act = act_layer(inplace=True)

    def forward(self, x):
        """Run forward pass."""
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        scale_factor=2.0,
        act_layer=nn.ReLU,
        norm_layer=nn.BatchNorm2d,
    ):
        super().__init__()
        conv_args = dict(kernel_size=3, padding=1, act_layer=act_layer)
        self.scale_factor = scale_factor
        if norm_layer is None:
            self.conv1 = Conv2dBnAct(in_channels, out_channels, **conv_args)
            self.conv2 = Conv2dBnAct(out_channels, out_channels, **conv_args)
        else:
            self.conv1 = Conv2dBnAct(
                in_channels, out_channels, norm_layer=norm_layer, **conv_args
            )
            self.conv2 = Conv2dBnAct(
                out_channels, out_channels, norm_layer=norm_layer, **conv_args
            )

    def forward(self, x, skip: Optional[torch.Tensor] = None):
        """Run forward pass."""
        if self.scale_factor != 1.0:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels=(256, 128, 64, 32, 16),
        final_channels=1,
        norm_layer=nn.BatchNorm2d,
        center=False,
        scale_factors=None,
    ):
        super().__init__()

        if center:
            channels = encoder_channels[0]
            self.center = DecoderBlock(
                channels, channels, scale_factor=1.0, norm_layer=norm_layer
            )
        else:
            self.center = nn.Identity()

        in_channels = [
            in_chs + skip_chs
            for in_chs, skip_chs in zip(
                [encoder_channels[0]] + list(decoder_channels[:-1]),
                list(encoder_channels[1:]) + [0],
            )
        ]
        out_channels = decoder_channels

        self.blocks = nn.ModuleList()
        if scale_factors is None:
            scale_factors = (2,) * len(out_channels)
        for in_chs, out_chs, scale_fct in zip(in_channels, out_channels, scale_factors):
            self.blocks.append(
                DecoderBlock(
                    in_chs,
                    out_chs,
                    scale_factor=scale_fct,
                    norm_layer=norm_layer,
                )
            )
        self.final_conv = nn.Conv2d(
            out_channels[-1], final_channels, kernel_size=(1, 1)
        )

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x: List[torch.Tensor]):
        """Run forward pass."""
        encoder_head = x[0]
        skips = x[1:]
        x = self.center(encoder_head)
        for i, b in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = b(x, skip)
        x = self.final_conv(x)
        return x
