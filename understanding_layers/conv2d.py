import torch
import math
import torch

def _pair(v):
    if isinstance(v, tuple):
        return v
    return (v, v)

import torch
import math

def _pair(v):
    if isinstance(v, tuple):
        return v
    return (v, v)

def conv2d_from_scratch(input, weight, bias=None, stride=1, padding=0):
    """
    input:  (N, C_in, H, W)
    weight: (C_out, C_in, K_h, K_w)
    bias:   (C_out,) or None
    stride: int or (stride_h, stride_w)
    padding: int or (pad_h, pad_w)

    No F.conv2d, no F.unfold, no F.pad, no as_strided.
    """
    device = input.device
    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)

    N, C_in, H, W = input.shape
    C_out, C_in_w, K_h, K_w = weight.shape
    assert C_in == C_in_w, "Input channels must match weight channels"

    # ---- 1) Manual zero padding ----
    if pad_h > 0 or pad_w > 0:
        H_p = H + 2 * pad_h
        W_p = W + 2 * pad_w
        input_padded = input.new_zeros((N, C_in, H_p, W_p))
        input_padded[:, :, pad_h:pad_h + H, pad_w:pad_w + W] = input
    else:
        input_padded = input
        H_p, W_p = H, W

    # ---- 2) Output spatial size ----
    H_out = (H_p - K_h) // stride_h + 1
    W_out = (W_p - K_w) // stride_w + 1
    if H_out <= 0 or W_out <= 0:
        raise ValueError(
            f"Output size <= 0 (H_out={H_out}, W_out={W_out}). "
            f"Check kernel_size={K_h,K_w}, padding={pad_h,pad_w}, stride={stride_h,stride_w}."
        )

    # ---- 3) Build indices for sliding windows ----
    h_out_idx = torch.arange(H_out, device=device)
    w_out_idx = torch.arange(W_out, device=device)
    kh_idx = torch.arange(K_h, device=device)
    kw_idx = torch.arange(K_w, device=device)

    # h = i * stride_h + kh, w = j * stride_w + kw
    h = h_out_idx[:, None, None, None] * stride_h + kh_idx[None, None, :, None]
    w = w_out_idx[None, :, None, None] * stride_w + kw_idx[None, None, None, :]

    # h, w broadcast to (H_out, W_out, K_h, K_w)

    # ---- 4) Gather patches: (N, C_in, H_out, W_out, K_h, K_w) ----
    windows = input_padded[:, :, h, w]

    # ---- 5) Reorder dims so flattening matches PyTorch's unfold ----
    # We want per patch vector order:
    #   for c in channels:
    #     for kh in kernel height:
    #       for kw in kernel width:
    #         value
    #
    # So go to (N, C_in, K_h, K_w, H_out, W_out) and then flatten.
    windows = windows.permute(0, 1, 4, 5, 2, 3).contiguous()
    N, C_in, K_h, K_w, H_out, W_out = windows.shape

    K = C_in * K_h * K_w
    L = H_out * W_out

    # (N, C_in, K_h, K_w, H_out, W_out) -> (N, K, L)
    cols = windows.view(N, K, L)  # (N, C_in*K_h*K_w, H_out*W_out)

    # ---- 6) Flatten weights in the same order ----
    weight_flat = weight.view(C_out, K)  # (C_out, K)

    # ---- 7) Batched matmul: out[n] = weight_flat @ cols[n] ----
    out = torch.einsum("oc,ncl->nol", weight_flat, cols)  # (N, C_out, L)

    # ---- 8) Reshape to (N, C_out, H_out, W_out) ----
    out = out.view(N, C_out, H_out, W_out)

    if bias is not None:
        out = out + bias.view(1, -1, 1, 1)

    return out


import torch
from torch import nn
import math

class MyConv2d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1, padding=0, bias=True):
        super().__init__()

        k_h, k_w = _pair(kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (k_h, k_w)
        self.stride = _pair(stride)
        self.padding = _pair(padding)

        # weight: (C_out, C_in, K_h, K_w)
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, k_h, k_w)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None

        # Kaiming-like init
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = in_channels * k_h * k_w
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return conv2d_from_scratch(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
        )
    

if __name__ == "__main__":
    import torch
    import torch.nn as nn

    N, C_in, H, W = 2, 3, 7, 7
    C_out = 4
    kernel_size = 3
    stride = 2
    padding = 1

    x = torch.randn(N, C_in, H, W)

    conv_ref = nn.Conv2d(C_in, C_out, kernel_size,
                         stride=stride, padding=padding, bias=True)
    y_ref = conv_ref(x)

    y_mine = conv2d_from_scratch(x, conv_ref.weight, conv_ref.bias,
                                 stride=stride, padding=padding)

    print("max abs diff:", (y_ref - y_mine).abs().max().item())
