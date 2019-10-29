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


class Choice_Block(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, k_size, stride):
        super(Choice_Block, self).__init__()
        padding = k_size // 2
        self.padding = padding
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.k_size = k_size
        self.stride = stride

        # conv_1x1
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)
        # depthwise
        self.DW_conv1 = nn.Conv2d(mid_channels, mid_channels, kernel_size=k_size,
                                  stride=stride, padding=padding, groups=mid_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        # pointwise
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        if stride == 2:
            # depthwise
            self.DW_conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=k_size,
                                      stride=2, padding=padding, groups=in_channels, bias=False)
            self.bn4 = nn.BatchNorm2d(in_channels)
            # pointwise
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn5 = nn.BatchNorm2d(out_channels)
            self.relu3 = nn.ReLU(inplace=True)

        # for p in self.parameters():
        #     print(p)
        #     p.requires_grad = False
    def gradient(self, flag):
        for p in self.parameters():
            print(p)
            p.requires_grad = flag

    def forward(self, x):
        # self.stride == 1
        if self.stride == 1:
            x1, x2 = channel_split(x, self.in_channels)
        # self.stride == 2
        else:
            # depthwise
            x1 = self.DW_conv2(x)
            x1 = self.bn4(x1)
            # pointwise
            x1 = self.conv3(x1)
            x1 = self.bn5(x1)
            x1 = self.relu3(x1)
            x2 = x
        # conv_1x1
        x2 = self.conv1(x2)
        x2 = self.bn1(x2)
        x2 = self.relu1(x2)
        # depthwise
        x2 = self.DW_conv1(x2)
        x2 = self.bn2(x2)
        # pointwise
        x2 = self.conv2(x2)
        x2 = self.bn3(x2)
        x2 = self.relu2(x2)
        # channel_shuffle
        x3 = torch.cat((x1, x2), 1)
        x3 = shuffle_channels(x3)

        return x3


class Choice_Block_x(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride):
        super(Choice_Block_x, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.stride = stride

        # depthwise
        self.DW_conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                  stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        # pointwise
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)
        # depthwise
        self.DW_conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3,
                                  stride=1, padding=1, groups=mid_channels, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_channels)
        # pointwise
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(mid_channels)
        self.relu2 = nn.ReLU(inplace=True)
        # depthwise
        self.DW_conv3 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3,
                                  stride=1, padding=1, groups=mid_channels, bias=False)
        self.bn5 = nn.BatchNorm2d(mid_channels)
        # pointwise
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn6 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)

        if stride == 2:
            # depthwise
            self.DW_conv4 = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                      stride=2, padding=1, groups=in_channels, bias=False)
            self.bn7 = nn.BatchNorm2d(in_channels)
            # pointwise
            self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn8 = nn.BatchNorm2d(out_channels)
            self.relu4 = nn.ReLU(inplace=True)

    def gradient(self, flag):
        for p in self.parameters():
            print(p)
            p.requires_grad = flag

    def forward(self, x):
        # self.stride == 1
        if self.stride == 1:
            x1, x2 = channel_split(x, self.in_channels)
        # self.stride == 2
        else:
            # depthwise
            x1 = self.DW_conv4(x)
            x1 = self.bn7(x1)
            # pointwise
            x1 = self.conv4(x1)
            x1 = self.bn8(x1)
            x1 = self.relu4(x1)
            x2 = x
        # depthwise
        x2 = self.DW_conv1(x2)
        x2 = self.bn1(x2)
        # pointwise
        x2 = self.conv1(x2)
        x2 = self.bn2(x2)
        x2 = self.relu1(x2)
        # depthwise
        x2 = self.DW_conv2(x2)
        x2 = self.bn3(x2)
        # pointwise
        x2 = self.conv2(x2)
        x2 = self.bn4(x2)
        x2 = self.relu2(x2)
        # depthwise
        x2 = self.DW_conv3(x2)
        x2 = self.bn5(x2)
        # pointwise
        x2 = self.conv3(x2)
        x2 = self.bn6(x2)
        x2 = self.relu3(x2)
        # channel_shuffle
        x3 = torch.cat((x1, x2), 1)
        x3 = shuffle_channels(x3)

        return x3



