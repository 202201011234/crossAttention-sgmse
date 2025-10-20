# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Layers for defining NCSN++.
"""
from . import layers
from . import up_or_down_sampling
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

conv1x1 = layers.ddpm_conv1x1
conv3x3 = layers.ddpm_conv3x3
NIN = layers.NIN
default_init = layers.default_init


class GaussianFourierProjection(nn.Module):
  """Gaussian Fourier embeddings for noise levels."""

  def __init__(self, embedding_size=256, scale=1.0):
    super().__init__()
    self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Combine(nn.Module):
  """Combine information from skip connections."""

  def __init__(self, dim1, dim2, method='cat'):
    super().__init__()
    self.dim1 = dim1
    self.dim2 = dim2
    self.method = method
    # 不在这里创建卷积层，而是在forward方法中动态创建

  def forward(self, x, y):
    # 动态计算实际的输入通道数
    actual_dim1 = x.shape[1]
    actual_dim2 = y.shape[1]
    
    # 如果实际通道数与初始化时的通道数不同，需要动态创建卷积层
    if actual_dim1 != self.dim1 or actual_dim2 != self.dim2:
      # 动态创建新的卷积层，使用实际的通道数
      self.Conv_0 = conv1x1(actual_dim1, actual_dim2)
      self.Conv_0 = self.Conv_0.to(x.device)
      h = self.Conv_0(x)
    else:
      # 如果通道数匹配，使用预定义的卷积层（如果存在）
      if hasattr(self, 'Conv_0'):
        h = self.Conv_0(x)
      else:
        # 第一次调用时创建卷积层
        self.Conv_0 = conv1x1(actual_dim1, actual_dim2)
        self.Conv_0 = self.Conv_0.to(x.device)
        h = self.Conv_0(x)
        
    if self.method == 'cat':
      return torch.cat([h, y], dim=1)
    elif self.method == 'sum':
      return h + y
    else:
      raise ValueError(f'Method {self.method} not recognized.')


class AttnBlockpp(nn.Module):
  """Channel-wise self-attention block. Modified from DDPM."""

  def __init__(self, channels, skip_rescale=False, init_scale=0.):
    super().__init__()
    self.GroupNorm_0 = nn.GroupNorm(num_groups=min(channels // 4, 32), num_channels=channels,
                                  eps=1e-6)
    self.NIN_0 = NIN(channels, channels)
    self.NIN_1 = NIN(channels, channels)
    self.NIN_2 = NIN(channels, channels)
    self.NIN_3 = NIN(channels, channels, init_scale=init_scale)
    self.skip_rescale = skip_rescale

  def forward(self, x):
    B, C, H, W = x.shape
    h = self.GroupNorm_0(x)
    q = self.NIN_0(h)
    k = self.NIN_1(h)
    v = self.NIN_2(h)

    w = torch.einsum('bchw,bcij->bhwij', q, k) * (int(C) ** (-0.5))
    w = torch.reshape(w, (B, H, W, H * W))
    w = F.softmax(w, dim=-1)
    w = torch.reshape(w, (B, H, W, H, W))
    h = torch.einsum('bhwij,bcij->bchw', w, v)
    h = self.NIN_3(h)
    if not self.skip_rescale:
      return x + h
    else:
      return (x + h) / np.sqrt(2.)


class Upsample(nn.Module):
  def __init__(self, in_ch=None, out_ch=None, with_conv=False, fir=False,
               fir_kernel=(1, 3, 3, 1)):
    super().__init__()
    out_ch = out_ch if out_ch else in_ch
    if not fir:
      if with_conv:
        self.Conv_0 = conv3x3(in_ch, out_ch)
    else:
      if with_conv:
        self.Conv2d_0 = up_or_down_sampling.Conv2d(in_ch, out_ch,
                                                 kernel=3, up=True,
                                                 resample_kernel=fir_kernel,
                                                 use_bias=True,
                                                 kernel_init=default_init())
    self.fir = fir
    self.with_conv = with_conv
    self.fir_kernel = fir_kernel
    self.out_ch = out_ch

  def forward(self, x):
    B, C, H, W = x.shape
    if not self.fir:
      h = F.interpolate(x, (H * 2, W * 2), 'nearest')
      if self.with_conv:
        h = self.Conv_0(h)
    else:
      if not self.with_conv:
        h = up_or_down_sampling.upsample_2d(x, self.fir_kernel, factor=2)
      else:
        h = self.Conv2d_0(x)

    return h


class Downsample(nn.Module):
  def __init__(self, in_ch=None, out_ch=None, with_conv=False, fir=False,
               fir_kernel=(1, 3, 3, 1)):
    super().__init__()
    out_ch = out_ch if out_ch else in_ch
    if not fir:
      if with_conv:
        self.Conv_0 = conv3x3(in_ch, out_ch, stride=2, padding=0)
    else:
      if with_conv:
        self.Conv2d_0 = up_or_down_sampling.Conv2d(in_ch, out_ch,
                                                 kernel=3, down=True,
                                                 resample_kernel=fir_kernel,
                                                 use_bias=True,
                                                 kernel_init=default_init())
    self.fir = fir
    self.fir_kernel = fir_kernel
    self.with_conv = with_conv
    self.out_ch = out_ch

  def forward(self, x):
    B, C, H, W = x.shape
    if not self.fir:
      if self.with_conv:
        x = F.pad(x, (0, 1, 0, 1))
        x = self.Conv_0(x)
      else:
        x = F.avg_pool2d(x, 2, stride=2)
    else:
      if not self.with_conv:
        x = up_or_down_sampling.downsample_2d(x, self.fir_kernel, factor=2)
      else:
        x = self.Conv2d_0(x)

    return x


class ResnetBlockDDPMpp(nn.Module):
  """ResBlock adapted from DDPM."""

  def __init__(self, act, in_ch, out_ch=None, temb_dim=None, conv_shortcut=False,
               dropout=0.1, skip_rescale=False, init_scale=0.):
    super().__init__()
    out_ch = out_ch if out_ch else in_ch
    self.GroupNorm_0 = nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6)
    self.Conv_0 = conv3x3(in_ch, out_ch)
    if temb_dim is not None:
      self.Dense_0 = nn.Linear(temb_dim, out_ch)
      self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
      nn.init.zeros_(self.Dense_0.bias)
    self.GroupNorm_1 = nn.GroupNorm(num_groups=min(out_ch // 4, 32), num_channels=out_ch, eps=1e-6)
    self.Dropout_0 = nn.Dropout(dropout)
    self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)
    if in_ch != out_ch:
      if conv_shortcut:
        self.Conv_2 = conv3x3(in_ch, out_ch)
      else:
        self.NIN_0 = NIN(in_ch, out_ch)

    self.skip_rescale = skip_rescale
    self.act = act
    self.out_ch = out_ch
    self.conv_shortcut = conv_shortcut

  def forward(self, x, temb=None):
    # 动态计算实际的输入通道数，而不是使用固定的初始化值
    actual_in_ch = x.shape[1]
    
    # 如果实际通道数与初始化时的通道数不同，需要动态创建GroupNorm层
    if actual_in_ch != self.GroupNorm_0.num_channels:
      # 动态创建新的GroupNorm层，使用实际的通道数
      self.GroupNorm_0 = nn.GroupNorm(num_groups=min(actual_in_ch // 4, 32), num_channels=actual_in_ch, eps=1e-6)
      self.GroupNorm_0 = self.GroupNorm_0.to(x.device)
      h = self.act(self.GroupNorm_0(x))
    else:
      h = self.act(self.GroupNorm_0(x))

    # 如果实际输入通道数与Conv_0层的输入通道数不匹配，需要动态调整卷积层
    if actual_in_ch != self.Conv_0.in_channels:
      # 动态创建新的卷积层，使用实际的输入通道数
      self.Conv_0 = conv3x3(actual_in_ch, self.out_ch)
      self.Conv_0 = self.Conv_0.to(x.device)
      # 同时更新Conv_2层的输入通道数
      if hasattr(self, 'Conv_2'):
        self.Conv_2 = conv1x1(actual_in_ch, self.out_ch)
        self.Conv_2 = self.Conv_2.to(x.device)
    
    h = self.Conv_0(h)
    # Add bias to each feature map conditioned on the time embedding
    if temb is not None:
      h += self.Dense_0(self.act(temb))[:, :, None, None]
    
    # 对于GroupNorm_1，使用固定的out_ch，因为这是输出通道数
    h = self.act(self.GroupNorm_1(h))
    h = self.Dropout_0(h)
    h = self.Conv_1(h)

    # 更新条件判断，使用实际的输入通道数
    if actual_in_ch != self.out_ch or self.up or self.down:
      # 如果Conv_2层不存在，动态创建它
      if not hasattr(self, 'Conv_2'):
        self.Conv_2 = conv1x1(actual_in_ch, self.out_ch)
        self.Conv_2 = self.Conv_2.to(x.device)
      x = self.Conv_2(x)

    if not self.skip_rescale:
      return x + h
    else:
      return (x + h) / np.sqrt(2.)


class ResnetBlockBigGANpp(nn.Module):
  def __init__(self, act, in_ch, out_ch=None, temb_dim=None, up=False, down=False,
               dropout=0.1, fir=False, fir_kernel=(1, 3, 3, 1),
               skip_rescale=True, init_scale=0.):
    super().__init__()

    out_ch = out_ch if out_ch else in_ch
    self.GroupNorm_0 = nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6)
    self.up = up
    self.down = down
    self.fir = fir
    self.fir_kernel = fir_kernel

    self.Conv_0 = conv3x3(in_ch, out_ch)
    if temb_dim is not None:
      self.Dense_0 = nn.Linear(temb_dim, out_ch)
      self.Dense_0.weight.data = default_init()(self.Dense_0.weight.shape)
      nn.init.zeros_(self.Dense_0.bias)

    self.GroupNorm_1 = nn.GroupNorm(num_groups=min(out_ch // 4, 32), num_channels=out_ch, eps=1e-6)
    self.Dropout_0 = nn.Dropout(dropout)
    self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)
    if in_ch != out_ch or up or down:
      self.Conv_2 = conv1x1(in_ch, out_ch)

    self.skip_rescale = skip_rescale
    self.act = act
    self.in_ch = in_ch
    self.out_ch = out_ch

  def forward(self, x, temb=None):
    # 动态计算实际的输入通道数，而不是使用固定的初始化值
    actual_in_ch = x.shape[1]
    
    # 如果实际通道数与初始化时的通道数不同，需要动态创建GroupNorm层
    if actual_in_ch != self.GroupNorm_0.num_channels:
      # 动态创建新的GroupNorm层，使用实际的通道数
      self.GroupNorm_0 = nn.GroupNorm(num_groups=min(actual_in_ch // 4, 32), num_channels=actual_in_ch, eps=1e-6)
      self.GroupNorm_0 = self.GroupNorm_0.to(x.device)
      h = self.act(self.GroupNorm_0(x))
    else:
      h = self.act(self.GroupNorm_0(x))

    if self.up:
      if self.fir:
        h = up_or_down_sampling.upsample_2d(h, self.fir_kernel, factor=2)
        x = up_or_down_sampling.upsample_2d(x, self.fir_kernel, factor=2)
      else:
        h = up_or_down_sampling.naive_upsample_2d(h, factor=2)
        x = up_or_down_sampling.naive_upsample_2d(x, factor=2)
    elif self.down:
      if self.fir:
        h = up_or_down_sampling.downsample_2d(h, self.fir_kernel, factor=2)
        x = up_or_down_sampling.downsample_2d(x, self.fir_kernel, factor=2)
      else:
        h = up_or_down_sampling.naive_downsample_2d(h, factor=2)
        x = up_or_down_sampling.naive_downsample_2d(x, factor=2)

    # 如果实际输入通道数与Conv_0层的输入通道数不匹配，需要动态调整卷积层
    if actual_in_ch != self.Conv_0.in_channels:
      # 动态创建新的卷积层，使用实际的输入通道数
      self.Conv_0 = conv3x3(actual_in_ch, self.out_ch)
      self.Conv_0 = self.Conv_0.to(x.device)
      # 同时更新Conv_2层的输入通道数
      if hasattr(self, 'Conv_2'):
        self.Conv_2 = conv1x1(actual_in_ch, self.out_ch)
        self.Conv_2 = self.Conv_2.to(x.device)
    
    h = self.Conv_0(h)
    # Add bias to each feature map conditioned on the time embedding
    if temb is not None:
      h += self.Dense_0(self.act(temb))[:, :, None, None]
    
    # 对于GroupNorm_1，使用固定的out_ch，因为这是输出通道数
    h = self.act(self.GroupNorm_1(h))
    h = self.Dropout_0(h)
    h = self.Conv_1(h)

    # 更新条件判断，使用实际的输入通道数
    if actual_in_ch != self.out_ch or self.up or self.down:
      # 如果Conv_2层不存在，动态创建它
      if not hasattr(self, 'Conv_2'):
        self.Conv_2 = conv1x1(actual_in_ch, self.out_ch)
        self.Conv_2 = self.Conv_2.to(x.device)
      x = self.Conv_2(x)

    if not self.skip_rescale:
      return x + h
    else:
      return (x + h) / np.sqrt(2.)
