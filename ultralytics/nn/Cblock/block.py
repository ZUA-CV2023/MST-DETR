# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Block modules."""

import torch
import torch.nn as nn
import numpy as np
from ..modules.dynamic_adapter import *
import torch.nn.functional as F
from collections import OrderedDict
from torchvision.ops import DeformConv2d
from ..modules.merge import *
from ..modules.conv import Conv, DWConv, GhostConv, LightConv, RepConv
from ..modules.transformer import TransformerBlock
from ..modules.shvit import SHSA
__all__ = ('DFL', 'HGBlock', 'HGStem', 'SPP', 'SPPF', 'C1', 'C2', 'C3', 'C2f', 'C3x', 'C3TR', 'C3Ghost',
           'GhostBottleneck', 'Bottleneck', 'BottleneckCSP', 'Proto', 'RepC3', 'ConvNormLayer', 'BasicBlock', 
           'BottleNeck', 'Blocks','SelectiveFusion',
           'Partial_conv3','PartialConvWithCBAMAttention',
           # 'PartialConvWithAttention'
           )


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391c
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck with 2 convolutions module with arguments ch_in, ch_out, number, shortcut,
        groups, expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1,
                                                                            act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))

################################### RT-DETR PResnet ###################################
def get_activation(act: str, inpace: bool=True):
    '''get activation
    '''
    act = act.lower()
    
    if act == 'silu':
        m = nn.SiLU()

    elif act == 'relu':
        m = nn.ReLU()

    elif act == 'leaky_relu':
        m = nn.LeakyReLU()

    elif act == 'silu':
        m = nn.SiLU()
    
    elif act == 'gelu':
        m = nn.GELU()
        
    elif act is None:
        m = nn.Identity()
    
    elif isinstance(act, nn.Module):
        m = act

    else:
        raise RuntimeError('')  

    if hasattr(m, 'inplace'):
        m.inplace = inpace
    
    return m 

class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in, 
            ch_out, 
            kernel_size, 
            stride, 
            padding=(kernel_size-1)//2 if padding is None else padding, 
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))
    
    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__()

        self.shortcut = shortcut

        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(ch_in, ch_out, 1, 1))
                ]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out, 1, stride)

        self.branch2a = ConvNormLayer(ch_in, ch_out, 3, stride, act=act)
        self.branch2b = ConvNormLayer(ch_out, ch_out, 3, 1, act=None)
        self.act = nn.Identity() if act is None else get_activation(act) 


    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        if self.shortcut:
            short = x
        else:
            short = self.short(x)
        
        out = out + short
        out = self.act(out)

        return out


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__()

        if variant == 'a':
            stride1, stride2 = stride, 1
        else:
            stride1, stride2 = 1, stride

        width = ch_out 

        self.branch2a = ConvNormLayer(ch_in, width, 1, stride1, act=act)
        self.branch2b = ConvNormLayer(width, width, 3, stride2, act=act)
        self.branch2c = ConvNormLayer(width, ch_out * self.expansion, 1, 1)

        self.shortcut = shortcut
        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(ch_in, ch_out * self.expansion, 1, 1))
                ]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out * self.expansion, 1, stride)

        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        out = self.branch2c(out)

        if self.shortcut:
            short = x
        else:
            short = self.short(x)

        out = out + short
        out = self.act(out)

        return out


class Blocks(nn.Module):
    def __init__(self, ch_in, ch_out, block, count, stage_num, act='relu', input_resolution=None, sr_ratio=None, kernel_size=None, kan_name=None, variant='d'):
        super().__init__()

        self.blocks = nn.ModuleList()
        for i in range(count):
            if input_resolution is not None and sr_ratio is not None:
                self.blocks.append(
                    block(
                        ch_in, 
                        ch_out,
                        stride=2 if i == 0 and stage_num != 2 else 1, 
                        shortcut=False if i == 0 else True,
                        variant=variant,
                        act=act,
                        input_resolution=input_resolution,
                        sr_ratio=sr_ratio)
                )
            elif kernel_size is not None:
                self.blocks.append(
                    block(
                        ch_in, 
                        ch_out,
                        stride=2 if i == 0 and stage_num != 2 else 1, 
                        shortcut=False if i == 0 else True,
                        variant=variant,
                        act=act,
                        kernel_size=kernel_size)
                )
            elif kan_name is not None:
                self.blocks.append(
                    block(
                        ch_in, 
                        ch_out,
                        stride=2 if i == 0 and stage_num != 2 else 1, 
                        shortcut=False if i == 0 else True,
                        variant=variant,
                        act=act,
                        kan_name=kan_name)
                )
            else:
                self.blocks.append(
                    block(
                        ch_in, 
                        ch_out,
                        stride=2 if i == 0 and stage_num != 2 else 1, 
                        shortcut=False if i == 0 else True,
                        variant=variant,
                        act=act)
                )
            if i == 0:
                ch_in = ch_out * block.expansion

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        return out


class SelectiveFusion(nn.Module):
    def __init__(self, input_dim, num_sub_layer = 4,device='cuda'):
        super(SelectiveFusion, self).__init__()
        self.token_select = TokenSelect(input_dim, num_sub_layer)
        # self.mapping_layer = nn.Linear(input_dim, input_dim)
        self.attention_layer = nn.Linear(input_dim * 2, 1)
        self.device = device
        self.to(self.device)

    def forward(self, feats):
        # Token 选择与映射
        token_select, logits = self.token_select(feats)
        token_select = token_select.to(self.device)
        logits = logits.to(self.device)

        # 映射 token
        # token_select_mapped = self.mapping_layer(token_select.size(-1),token_select.size(-1))
        # 创建映射层，确保其输入维度与 token_select 的最后一个维度匹配
        mapping_layer = nn.Linear(token_select.size(-1), feats.size(-1)).to(self.device)
        token_select_mapped = mapping_layer(token_select)
        # bipartite_soft_matching 合并 token
        merge_func, _ = bipartite_soft_matching(logits, r=token_select.size(1) // 2)
        feats_merged = merge_func(feats)

        # 插值特征
        upsampled_feats = F.interpolate(feats_merged.unsqueeze(0), size=(8400, feats.size(-1)), mode='bilinear',
                                        align_corners=True).squeeze(0)

        # 计算注意力权重
        combined_features = torch.cat((token_select_mapped, upsampled_feats), dim=-1)
        attention_weights = torch.sigmoid(self.attention_layer(combined_features)).squeeze(-1)
        attention_weights = attention_weights.unsqueeze(-1)

        # 动态加权组合
        combined_feats = (attention_weights * feats * token_select_mapped) + (
                    (1 - attention_weights) * (feats * upsampled_feats))

        return combined_feats
############Partial_conv3 START#############
'''
        class Partial_conv3(nn.Module):

    def __init__(self, dim,output_dim , n_div, forward='split_cat', reduce_channels=True):
        super().__init__()
        self.dim_conv3 = dim // n_div  # 部分通道进行卷积
        self.dim_untouched = dim - self.dim_conv3  # 未处理的通道
        self.reduce_channels = reduce_channels  # 控制是否减少输出通道数
        self.output_dim = output_dim  # 新增参数，指定输出的通道数

        # 定义卷积操作，只对部分通道进行
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

        # 若指定了 output_dim，则定义调整通道数的卷积层
        if self.output_dim is not None and self.output_dim != dim:
            self.channel_adjust = nn.Conv2d(dim, self.output_dim, 1, 1, bias=False)
        else:
            self.channel_adjust = None

    def forward_split_cat(self, x):
        # 分割通道：x1 是卷积的部分，x2 是未卷积的部分
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)

        # 只对 x1 进行卷积
        x1 = self.partial_conv3(x1)

        # 控制拼接时是否减少通道数
        if self.reduce_channels:
            x = x1  # 只保留卷积后的部分
        else:
            # 拼接卷积后的 x1 和未卷积的部分
            x = torch.cat((x1, x2), dim=1)

        # 如果需要调整通道数，则应用调整层
        if self.channel_adjust is not None:
            x = self.channel_adjust(x)

        return x

    def forward_slicing(self, x):
        # 只对部分通道进行卷积
        x = x.clone()  # 保留原始输入以便后续残差连接
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        # 如果需要调整通道数，则应用调整层
        if self.channel_adjust is not None:
            x = self.channel_adjust(x)

        return x
'''
class TokenSelection(nn.Module):
    def __init__(self, num_tokens):
        super(TokenSelection, self).__init__()

    def forward(self, x):
        # x 的形状为 (B, C, H, W)
        B, C, H, W = x.shape
        # 动态计算 num_tokens 为 H * W 的一半
        num_tokens = (H * W) // 2
        # 将输入变换为 (B, C, H*W)
        input_flattened = x.reshape(B, C, -1)  # (B, C, H*W)

        # 计算权重矩阵
        weight_matrix = torch.bmm(input_flattened.permute(0, 2, 1), input_flattened)  # (B, H*W, H*W)

        # 归一化权重矩阵
        weight_matrix_normalized = F.softmax(weight_matrix, dim=-1)  # 对每个 H*W 归一化

        # 计算每个 token 的权重
        token_weights = weight_matrix_normalized.mean(dim=-1)  # (B, H*W)

        # 选择最大的 token 的索引
        _, top_indices = torch.topk(token_weights, num_tokens, dim=-1)  # (B, num_tokens)

        # 将 top_indices 的 token 选择出来 (X1)，其余的为 X2
        X1 = torch.gather(input_flattened, 2,
                          top_indices.unsqueeze(1).expand(B, C, num_tokens))  # (B, C, num_tokens)

        # 通过掩码计算剩余 token（X2），即剩下的 H*W - num_tokens
        all_indices = torch.arange(H * W, device=x.device).unsqueeze(0).expand(B, -1)  # (B, H*W)
        mask = torch.ones_like(all_indices, dtype=torch.bool)  # 初始全为 True
        mask.scatter_(1, top_indices, False)  # 将选择的 token 索引位置设为 False

        # 通过掩码选择剩下的 token 作为 X2-
        remaining_indices = all_indices[mask].view(B, -1)  # (B, H*W - num_tokens)
        X2 = torch.gather(input_flattened, 2, remaining_indices.unsqueeze(1).expand(B, C, remaining_indices.size(
            1)))  # (B, C, H*W - num_tokens)

        # 还原 X1 和 X2 的形状为 (B, C, H // 2, W // 2)
        # new_H, new_W = int(num_tokens ** 0.5), int(num_tokens ** 0.5)  # 动态计算高度和宽度
        X1 = X1.reshape(B, C, H, int(num_tokens/W))  # 动态恢复 X1 的形状 (B, C, new_H, new_W)
        X2 = X2.reshape(B, C, H, int(num_tokens/W))  # 动态恢复 X2 的形状 (B, C, H - new_H, W - new_W)
        # # 还原 X1 和 X2 的形状回到 (B, C, H, W)恢复的大小应该是原图的一半
        # X1 = X1.view(B, C, int((H*W) ** 0.5), int((H*W) ** 0.5))  # 恢复 X1 的形状
        # X2 = X2.view(B, C, int((H*W) ** 0.5), int((H*W) ** 0.5))  # 恢复 X2 的形状

        return X1, X2
class Partial_conv3(nn.Module):
    def __init__(self, c1, c2, n_div, forward, downsample=False):
        super().__init__()
        self.c1 = c1
        # self.dim_conv3 = c1 // n_div  # 输入的卷积部分
        # self.dim_untouched = c1 - self.dim_conv3  # 输入的不变部分

        # 使用可形变卷积替代部分卷积层
        # self.offset_conv = nn.Conv2d(self.dim_conv3, 18, kernel_size=3, padding=1)  # 用于计算offset
        self.offset_conv = nn.Conv2d(self.c1, 18, kernel_size=3, padding=1)  # 用于计算offset
        # 使用可形变卷积替代部分卷积层
        # self.partial_conv3 = DeformConv2d(self.dim_conv3, c2, 3, padding=1, bias=False)
        self.partial_conv3 = DeformConv2d(self.c1, c2, 3, padding=1, bias=False)
        # 使用深度卷积处理不变部分
        # # 这里使用深度可分离卷积的实现
        # self.depthwise_conv = nn.Conv2d(self.dim_untouched, self.dim_untouched, 3, padding=1, groups=self.dim_untouched, bias=False)
        # self.pointwise_conv = nn.Conv2d(self.dim_untouched, c2, 1, bias=False)  # pointwise 卷积

        # 其他卷积类型的实现
        # self.standard_conv = nn.Conv2d(self.dim_untouched, c2, kernel_size=3, padding=1, bias=False)
        # self.group_conv = nn.Conv2d(self.dim_untouched, c2, kernel_size=3, padding=1, groups=2, bias=False)
        self.dilated_conv = nn.Conv2d(c1, c2, kernel_size=3, padding=2, dilation=2, bias=False)
        # 添加一个线性层用于将不变部分转换为合适的通道数
        # self.conv_to_c2 = nn.Conv2d(self.dim_unt
        # 使用线性变换 (1x1 卷积) 改变通道数
        # self.linear_transform = nn.Conv2d(self.dim_untouched, c2, 1, bias=False)
        # self.linear_transform = nn.Conv2d(self.c1, c2, 1, bias=False)
        # 初始化 token 选择模块
        self.token_selection = TokenSelection(num_tokens=None)  # 根据需要选择的 token 数量

        # 控制是否下采样
        if downsample:
            self.downsample_layer = nn.AvgPool2d(2)
        else:
            self.downsample_layer = nn.Identity()

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        # 推理时，保持原始输入不变
        # x = x.clone()
        # offset = self.offset_conv(x[:, :self.dim_conv3, :, :])
        # x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :], offset)
        # return x
        x = x.clone()
        offset = self.offset_conv(x[:, :self.c1, :, :])
        x[:, :self.c1, :, :] = self.partial_conv3(x[:, :self.c1, :, :], offset)
        return x

    def forward_split_cat(self, x):
        # # 将输入分成卷积部分和不变部分
        # x1 = x[:, :self.dim_conv3, :, :]
        # x2 = x[:, self.dim_conv3:, :, :]
        # 将输入分成卷积部分和不变部分
        x1 ,x2 = self.token_selection(x)
        # 对 x1 进行可形变卷积
        offset = self.offset_conv(x1)
        x1 = self.partial_conv3(x1, offset)

        # # 对 x2 使用深度可分离卷积
        # x2 = self.depthwise_conv(x2)
        # x2 = self.pointwise_conv(x2)

        # 其他卷积类型可以手动开启
        # x2 = self.standard_conv(x2)  # 使用标准卷积
        # x2 = self.group_conv(x2)      # 使用组卷积
        x2 = self.dilated_conv(x2)    # 使用膨胀卷积
        # 使用1x1卷积将不变部分转换为c2通道
        # x2 = self.conv_to_c2(x2)  # 将x2转换为c2通道
        # 使用线性变换（1x1卷积）改变x2的通道数
        # x2 = self.linear_transform(x2)
        x2 = x2.permute(0, 1, 3, 2)  # 从 (B, C, H, W) 变为 (B, C, W, H)
        # 合并两个部分
        x = torch.matmul(x1, x2)

        # 执行下采样（如果设置了 downsample）
        x = self.downsample_layer(x)
        return x

#################Partial_conv3 END ###############


############PartialConvWithCBAMAttention START#############
class TokenSelection(nn.Module):
    def __init__(self, num_tokens):
        super(TokenSelection, self).__init__()

    def forward(self, x):
        # x 的形状为 (B, C, H, W)
        B, C, H, W = x.shape
        # 动态计算 num_tokens 为 H * W 的一半
        num_tokens = (H * W) // 2
        # 将输入变换为 (B, C, H*W)
        input_flattened = x.reshape(B, C, -1)  # (B, C, H*W)

        # 计算权重矩阵
        weight_matrix = torch.bmm(input_flattened.permute(0, 2, 1), input_flattened)  # (B, H*W, H*W)

        # 归一化权重矩阵
        weight_matrix_normalized = F.softmax(weight_matrix, dim=-1)  # 对每个 H*W 归一化

        # 计算每个 token 的权重
        token_weights = weight_matrix_normalized.mean(dim=-1)  # (B, H*W)

        # 选择最大的 token 的索引
        _, top_indices = torch.topk(token_weights, num_tokens, dim=-1)  # (B, num_tokens)

        # 将 top_indices 的 token 选择出来 (X1)，其余的为 X2
        X1 = torch.gather(input_flattened, 2,
                          top_indices.unsqueeze(1).expand(B, C, num_tokens))  # (B, C, num_tokens)

        # 通过掩码计算剩余 token（X2），即剩下的 H*W - num_tokens
        all_indices = torch.arange(H * W, device=x.device).unsqueeze(0).expand(B, -1)  # (B, H*W)
        mask = torch.ones_like(all_indices, dtype=torch.bool)  # 初始全为 True
        mask.scatter_(1, top_indices, False)  # 将选择的 token 索引位置设为 False

        # 通过掩码选择剩下的 token 作为 X2-
        remaining_indices = all_indices[mask].view(B, -1)  # (B, H*W - num_tokens)
        X2 = torch.gather(input_flattened, 2, remaining_indices.unsqueeze(1).expand(B, C, remaining_indices.size(
            1)))  # (B, C, H*W - num_tokens)

        # 还原 X1 和 X2 的形状为 (B, C, H // 2, W // 2)
        # new_H, new_W = int(num_tokens ** 0.5), int(num_tokens ** 0.5)  # 动态计算高度和宽度
        X1 = X1.reshape(B, C, H, int(num_tokens/W))  # 动态恢复 X1 的形状 (B, C, new_H, new_W)
        X2 = X2.reshape(B, C, H, int(num_tokens/W))  # 动态恢复 X2 的形状 (B, C, H - new_H, W - new_W)
        # # 还原 X1 和 X2 的形状回到 (B, C, H, W)恢复的大小应该是原图的一半
        # X1 = X1.view(B, C, int((H*W) ** 0.5), int((H*W) ** 0.5))  # 恢复 X1 的形状
        # X2 = X2.view(B, C, int((H*W) ** 0.5), int((H*W) ** 0.5))  # 恢复 X2 的形状

        return X1, X2
class PartialConvWithCBAMAttention(nn.Module):
    def __init__(self, c1, c2, n_div, forward, downsample=False):
        super().__init__()
        self.c1 = c1
        # self.dim_conv3 = c1 // n_div  # 输入的卷积部分
        # self.dim_untouched = c1 - self.dim_conv3  # 输入的不变部分
        self.CBAM = CBAMLayer()
        # 使用可形变卷积替代部分卷积层
        # self.offset_conv = nn.Conv2d(self.dim_conv3, 18, kernel_size=3, padding=1)  # 用于计算offset
        self.offset_conv = nn.Conv2d(self.c1, 18, kernel_size=3, padding=1)  # 用于计算offset
        # 使用可形变卷积替代部分卷积层
        # self.partial_conv3 = DeformConv2d(self.dim_conv3, c2, 3, padding=1, bias=False)
        self.partial_conv3 = DeformConv2d(self.c1, c2, 3, padding=1, bias=False)
        # 使用深度卷积处理不变部分
        # # 这里使用深度可分离卷积的实现
        # self.depthwise_conv = nn.Conv2d(self.dim_untouched, self.dim_untouched, 3, padding=1, groups=self.dim_untouched, bias=False)
        # self.pointwise_conv = nn.Conv2d(self.dim_untouched, c2, 1, bias=False)  # pointwise 卷积

        # 其他卷积类型的实现
        # self.standard_conv = nn.Conv2d(self.dim_untouched, c2, kernel_size=3, padding=1, bias=False)
        # self.group_conv = nn.Conv2d(self.dim_untouched, c2, kernel_size=3, padding=1, groups=2, bias=False)
        self.dilated_conv = nn.Conv2d(c1, c2, kernel_size=3, padding=2, dilation=2, bias=False)
        # 添加一个线性层用于将不变部分转换为合适的通道数
        # self.conv_to_c2 = nn.Conv2d(self.dim_unt
        # 使用线性变换 (1x1 卷积) 改变通道数
        # self.linear_transform = nn.Conv2d(self.dim_untouched, c2, 1, bias=False)
        # self.linear_transform = nn.Conv2d(self.c1, c2, 1, bias=False)
        # 初始化 token 选择模块
        self.token_selection = TokenSelection(num_tokens=None)  # 根据需要选择的 token 数量

        # 控制是否下采样
        if downsample:
            self.downsample_layer = nn.AvgPool2d(2)
        else:
            self.downsample_layer = nn.Identity()

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        # 推理时，保持原始输入不变
        # x = x.clone()
        # offset = self.offset_conv(x[:, :self.dim_conv3, :, :])
        # x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :], offset)
        # return x
        x = x.clone()
        offset = self.offset_conv(x[:, :self.c1, :, :])
        x[:, :self.c1, :, :] = self.partial_conv3(x[:, :self.c1, :, :], offset)
        return x

    def forward_split_cat(self, x):
        # # 将输入分成卷积部分和不变部分
        # x1 = x[:, :self.dim_conv3, :, :]
        # x2 = x[:, self.dim_conv3:, :, :]
        # 将输入分成卷积部分和不变部分
        x1 ,x2 = self.token_selection(x)
        ###使用CBAM
        x1 = self.CBAM(x1)
        x2 = self.CBAM(x2)
        return x1,x2

        # # 对 x1 进行可形变卷积
        # offset = self.offset_conv(x1)
        # x1 = self.partial_conv3(x1, offset)

        # # 对 x2 使用深度可分离卷积
        # x2 = self.depthwise_conv(x2)
        # x2 = self.pointwise_conv(x2)

        # 其他卷积类型可以手动开启
        # x2 = self.standard_conv(x2)  # 使用标准卷积
        # x2 = self.group_conv(x2)      # 使用组卷积
        # x2 = self.dilated_conv(x2)    # 使用膨胀卷积
        # 使用1x1卷积将不变部分转换为c2通道
        # x2 = self.conv_to_c2(x2)  # 将x2转换为c2通道
        # 使用线性变换（1x1卷积）改变x2的通道数
        # x2 = self.linear_transform(x2)
        # x2 = x2.permute(0, 1, 3, 2)  # 从 (B, C, H, W) 变为 (B, C, W, H)
        # 合并两个部分
        # x = torch.matmul(x1, x2)

        # 执行下采样（如果设置了 downsample）
        x = self.downsample_layer(x)
        return x
class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x