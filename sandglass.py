import torch
import torch.nn.functional as F
from typing import Callable


class DepthwiseSeperableConvolutionalBlock(torch.nn.Module):
    def __init__(self, in_channels, kernel_size, out_channels, padding, stride, activation: Callable | None = None, bias: bool =True) -> None:
        super().__init__()
        self.activation = activation
        self.depthwise_layer = torch.nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, groups=in_channels, padding=padding, stride=stride, bias=bias)
        self.pointwise_layer = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias)

        self.batch_normalization_a = torch.nn.BatchNorm2d(num_features=in_channels)
        self.batch_normalization_b = torch.nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        depthwise_output = self.batch_normalization_a(self.depthwise_layer(x))
        if self.activation:
            pointwise_output = self.activation(self.batch_normalization_b(self.pointwise_layer(depthwise_output)))
        else:
            pointwise_output = self.batch_normalization_b(self.pointwise_layer(depthwise_output))
        return pointwise_output

class SandglassBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size, reduction_factor, bias=True) -> None:
        super().__init__()
        self.depthwise_block_a = DepthwiseSeperableConvolutionalBlock(in_channels=in_channels, kernel_size=kernel_size, out_channels=in_channels, padding=1, stride=1, bias=bias, activation=F.relu6)
        self.reduction_pointwise_conv_layer = torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels//reduction_factor, kernel_size=1, stride=1, padding=0, bias=bias)
        self.expansion_pointwise_conv_layer = torch.nn.Conv2d(in_channels=in_channels//reduction_factor, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.depthwise_block_b = DepthwiseSeperableConvolutionalBlock(in_channels=out_channels, kernel_size=kernel_size, out_channels=out_channels, padding=1, stride=stride, bias=bias)

        self.shortcut: bool = in_channels == out_channels
        self.batch_normalization_reduction = torch.nn.BatchNorm2d(num_features=in_channels//reduction_factor)
        self.batch_normalization_expansion = torch.nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        residual = x
        depthwise_a = self.depthwise_block_a(x)
        reduction = self.batch_normalization_reduction(self.reduction_pointwise_conv_layer(depthwise_a))
        expansion = F.relu6(self.batch_normalization_expansion(self.expansion_pointwise_conv_layer(reduction)))
        depthwise_b = self.depthwise_block_b(expansion)
        if self.shortcut:
            depthwise_b = depthwise_b + residual
        return depthwise_b

sandglass_block = SandglassBlock(32, 96, 2, 3, 2)

input_tensor = torch.rand(32, 32, 112, 112)

sandglass_output = sandglass_block(input_tensor)
print(sandglass_output.size())