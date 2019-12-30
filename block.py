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
    def __init__(self, in_channels, out_channels, kernel, stride, supernet=True):
        super(Choice_Block, self).__init__()
        padding = kernel // 2
        if supernet:
            self.affine = False
        else:
            self.affine = True
        self.stride = stride
        self.in_channels = in_channels
        self.mid_channels = out_channels // 2
        self.out_channels = out_channels - in_channels

        self.cb_main = nn.Sequential(
            # pw
            nn.Conv2d(self.in_channels, self.mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.mid_channels, affine=self.affine),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=kernel, stride=stride, padding=padding,
                      bias=False, groups=self.mid_channels),
            nn.BatchNorm2d(self.mid_channels, affine=self.affine),
            # pw_linear
            nn.Conv2d(self.mid_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.out_channels, affine=self.affine),
            nn.ReLU(inplace=True)
        )
        if stride == 2:
            self.cb_proj = nn.Sequential(
                # dw
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size=kernel, stride=2, padding=padding,
                          bias=False, groups=self.in_channels),
                nn.BatchNorm2d(self.in_channels, affine=self.affine),
                # pw
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.in_channels, affine=self.affine),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = channel_split(x, self.in_channels)
            y = torch.cat((self.cb_main(x1), x2), 1)
        else:
            y = torch.cat((self.cb_main(x), self.cb_proj(x)), 1)
        return y


class Choice_Block_x(nn.Module):
    def __init__(self, in_channels, out_channels, stride, supernet=True):
        super(Choice_Block_x, self).__init__()
        if supernet:
            self.affine = False
        else:
            self.affine = True
        self.stride = stride
        self.in_channels = in_channels
        self.mid_channels = out_channels // 2
        self.out_channels = out_channels - in_channels

        self.cb_main = nn.Sequential(
            # dw
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=stride,
                      padding=1, bias=False, groups=self.in_channels),
            nn.BatchNorm2d(self.in_channels, affine=self.affine),
            # pw
            nn.Conv2d(self.in_channels, self.mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.mid_channels, affine=self.affine),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1,
                      padding=1, bias=False, groups=self.mid_channels),
            nn.BatchNorm2d(self.mid_channels, affine=self.affine),
            # pw
            nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.mid_channels, affine=self.affine),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1,
                      padding=1, bias=False, groups=self.mid_channels),
            nn.BatchNorm2d(self.mid_channels, affine=self.affine),
            # pw
            nn.Conv2d(self.mid_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.out_channels, affine=self.affine),
            nn.ReLU(inplace=True)
        )
        if stride == 2:
            self.cb_proj = nn.Sequential(
                # dw
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=2,
                          padding=1, groups=self.in_channels, bias=False),
                nn.BatchNorm2d(self.in_channels, affine=self.affine),
                # pw
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.in_channels, affine=self.affine),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = channel_split(x, self.in_channels)
            y = torch.cat((self.cb_main(x1), x2), 1)
        else:
            y = torch.cat((self.cb_main(x), self.cb_proj(x)), 1)
        return y
