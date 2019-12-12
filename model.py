import torch.nn as nn
from block import Choice_Block, Choice_Block_x
import numpy as np


_repeat = [4, 4, 8, 4]
channel = [16, 64, 160, 320, 640]
last_channel = 1024


class Network(nn.Module):
    def __init__(self, classes=1000, gap_size=7):
        super(Network, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channel[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channel[0]),
            nn.ReLU6(inplace=True)
        )
        self.classes = classes
        self.gap_size = gap_size
        self.choice_block = nn.ModuleList([])
        for index, repeat in enumerate(_repeat):
            for i in range(repeat):
                if i == 0:
                    self.choice_3_2 = Choice_Block(channel[index], channel[index + 1] // 4,
                                                   channel[index + 1] // 2, k_size=3, stride=2)
                    self.choice_5_2 = Choice_Block(channel[index], channel[index + 1] // 4,
                                                   channel[index + 1] // 2, k_size=5, stride=2)
                    self.choice_7_2 = Choice_Block(channel[index], channel[index + 1] // 4,
                                                   channel[index + 1] // 2, k_size=7, stride=2)
                    self.choice_x_2 = Choice_Block_x(channel[index], channel[index + 1] // 4,
                                                     channel[index + 1] // 2, stride=2)
                    self.choice_block2 = nn.ModuleList(
                        [self.choice_3_2, self.choice_5_2, self.choice_7_2, self.choice_x_2])
                    self.choice_block.append(self.choice_block2)
                else:
                    self.choice_3_1 = Choice_Block(channel[index + 1] // 2, channel[index + 1] // 4,
                                                   channel[index + 1] // 2, k_size=3, stride=1)
                    self.choice_5_1 = Choice_Block(channel[index + 1] // 2, channel[index + 1] // 4,
                                                   channel[index + 1] // 2, k_size=5, stride=1)
                    self.choice_7_1 = Choice_Block(channel[index + 1] // 2, channel[index + 1] // 4,
                                                   channel[index + 1] // 2, k_size=7, stride=1)
                    self.choice_x_1 = Choice_Block_x(channel[index + 1] // 2, channel[index + 1] // 4,
                                                     channel[index + 1] // 2, stride=1)
                    self.choice_block1 = nn.ModuleList(
                        [self.choice_3_1, self.choice_5_1, self.choice_7_1, self.choice_x_1])
                    self.choice_block.append(self.choice_block1)
        self.last_conv = nn.Sequential(
            nn.Conv2d(channel[-1], last_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(last_channel),
            nn.ReLU6(inplace=True)
        )
        self.gap = nn.AvgPool2d(kernel_size=self.gap_size, stride=1, padding=0)
        self.fc = nn.Linear(last_channel, self.classes, bias=False)

    def forward(self, x, random=np.random.randint(4, size=20)):
        x = self.stem(x)
        # repeat
        for i, j in enumerate(random):
            x = self.choice_block[i][j](x)
        x = self.last_conv(x)
        x = self.gap(x)
        x = x.view(-1, last_channel)
        x = self.fc(x)
        return x
