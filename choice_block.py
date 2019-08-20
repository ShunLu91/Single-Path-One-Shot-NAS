import torch
import torch.nn as nn


def channel_split(x, split):
    """split a tensor into two pieces along channel dimension
    Args:
        x: input tensor
        split:(int) channel size for each pieces
    """
    assert x.size(1) == split * 2
    return torch.split(x, split, dim=1)


def shuffle_channels(x, groups=2):
    """shuffle channels of a 4-D Tensor"""

    batch_size, channels, height, width = x.size()
    assert channels % groups == 0
    channels_per_group = channels // groups
    # split into groups
    x = x.view(batch_size, groups, channels_per_group,
               height, width)
    # transpose 1, 2 axis
    x = x.transpose(1, 2).contiguous()
    # reshape into orignal
    x = x.view(batch_size, channels, height, width)
    return x


class Choice_Block3_1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Choice_Block3_1, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DW_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        x1, x2 = channel_split(x, self.in_channels)

        x2 = self.conv(x2)
        x2 = self.DW_conv(x2)
        x2 = self.conv(x2)

        x3 = torch.cat((x1, x2), 1)
        x3 = shuffle_channels(x3)
        return x3


class Choice_Block5_1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Choice_Block5_1, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DW_conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, groups=in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        x1, x2 = channel_split(x, self.in_channels)

        x2 = self.conv(x2)
        x2 = self.DW_conv(x2)
        x2 = self.conv(x2)

        x3 = torch.cat((x1, x2), 1)
        x3 = shuffle_channels(x3)
        return x3


class Choice_Block7_1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Choice_Block7_1, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DW_conv = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=1, padding=3, groups=in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        x1, x2 = channel_split(x, self.in_channels)

        x2 = self.conv(x2)
        x2 = self.DW_conv(x2)
        x2 = self.conv(x2)

        x3 = torch.cat((x1, x2), 1)
        x3 = shuffle_channels(x3)
        return x3


class Choice_Blockx_1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Choice_Blockx_1, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DW_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        x1, x2 = channel_split(x, self.in_channels)

        x2 = self.DW_conv(x2)
        x2 = self.conv(x2)
        x2 = self.DW_conv(x2)
        x2 = self.conv(x2)
        x2 = self.DW_conv(x2)
        x2 = self.conv(x2)

        x3 = torch.cat((x1, x2), 1)
        x3 = shuffle_channels(x3)
        return x3


class Choice_Block3_2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Choice_Block3_2, self).__init__()

        self.DW_conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, groups=in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.DW_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, groups=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        x1 = self.DW_conv1(x)
        x1 = self.conv1(x1)

        x2 = self.conv1(x)
        x2 = self.DW_conv2(x2)
        x2 = self.conv2(x2)

        x3 = torch.cat((x1, x2), 1)
        x3 = shuffle_channels(x3)
        return x3


class Choice_Block5_2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Choice_Block5_2, self).__init__()

        self.DW_conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=5, stride=2, padding=2, groups=in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.DW_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=2, padding=2, groups=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        x1 = self.DW_conv1(x)
        x1 = self.conv1(x1)

        x2 = self.conv1(x)
        x2 = self.DW_conv2(x2)
        x2 = self.conv2(x2)

        x3 = torch.cat((x1, x2), 1)
        x3 = shuffle_channels(x3)
        return x3


class Choice_Block7_2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Choice_Block7_2, self).__init__()

        self.DW_conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=7, stride=2, padding=3, groups=in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.DW_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=7, stride=2, padding=3, groups=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        x1 = self.DW_conv1(x)
        x1 = self.conv1(x1)

        x2 = self.conv1(x)
        x2 = self.DW_conv2(x2)
        x2 = self.conv2(x2)

        x3 = torch.cat((x1, x2), 1)
        x3 = shuffle_channels(x3)
        return x3


class Choice_Blockx_2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Choice_Blockx_2, self).__init__()

        self.DW_conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, groups=in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.DW_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        x1 = self.DW_conv1(x)
        x1 = self.conv1(x1)

        x2 = self.DW_conv1(x)
        x2 = self.conv1(x2)
        x2 = self.DW_conv2(x2)
        x2 = self.conv2(x2)
        x2 = self.DW_conv2(x2)
        x2 = self.conv2(x2)

        x3 = torch.cat((x1, x2), 1)
        x3 = shuffle_channels(x3)
        return x3
