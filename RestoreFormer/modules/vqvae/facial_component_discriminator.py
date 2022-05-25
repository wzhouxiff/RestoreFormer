import math
import random
import torch
from torch import nn
from torch.nn import functional as F

from basicsr.archs.stylegan2_arch import (ConvLayer, EqualConv2d, EqualLinear, ResBlock, ScaledLeakyReLU,
                                          StyleGAN2Generator)
from basicsr.ops.fused_act import FusedLeakyReLU
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class FacialComponentDiscriminator(nn.Module):

    def __init__(self):
        super(FacialComponentDiscriminator, self).__init__()

        self.conv1 = ConvLayer(3, 64, 3, downsample=False, resample_kernel=(1, 3, 3, 1), bias=True, activate=True)
        self.conv2 = ConvLayer(64, 128, 3, downsample=True, resample_kernel=(1, 3, 3, 1), bias=True, activate=True)
        self.conv3 = ConvLayer(128, 128, 3, downsample=False, resample_kernel=(1, 3, 3, 1), bias=True, activate=True)
        self.conv4 = ConvLayer(128, 256, 3, downsample=True, resample_kernel=(1, 3, 3, 1), bias=True, activate=True)
        self.conv5 = ConvLayer(256, 256, 3, downsample=False, resample_kernel=(1, 3, 3, 1), bias=True, activate=True)
        self.final_conv = ConvLayer(256, 1, 3, bias=True, activate=False)

    def forward(self, x, return_feats=False):
        feat = self.conv1(x)
        feat = self.conv3(self.conv2(feat))
        rlt_feats = []
        if return_feats:
            rlt_feats.append(feat.clone())
        feat = self.conv5(self.conv4(feat))
        if return_feats:
            rlt_feats.append(feat.clone())
        out = self.final_conv(feat)

        if return_feats:
            return out, rlt_feats
        else:
            return out, None
