import torch.nn as nn
from choice_block import Choice_Block, Choice_Block_x
import numpy as np


_repeat = [4, 4, 8, 4]
channel = [16, 64, 160, 320, 640]
# random = np.random.randint(4, size=20)

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # self.repeat = [4, 4, 8, 4]
        # self.channel = [16, 64, 160, 320, 640]

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        # self.choice_block = []
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
                    self.choice_block2 = nn.ModuleList([self.choice_3_2, self.choice_5_2, self.choice_7_2, self.choice_x_2])
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
                    self.choice_block1 = nn.ModuleList([self.choice_3_1, self.choice_5_1, self.choice_7_1, self.choice_x_1])
                    self.choice_block.append(self.choice_block1)

        self.conv2 = nn.Conv2d(640, 1024, kernel_size=1, stride=1, padding=0)
        self.gap = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
        self.fc = nn.Linear(1024, 10)

    def forward(self, x, random=np.random.randint(4, size=20)):

        x = self.conv1(x)
        # repeat 4
        x = self.choice_block[0][random[0]](x)
        x = self.choice_block[1][random[1]](x)
        x = self.choice_block[2][random[2]](x)
        x = self.choice_block[3][random[3]](x)

        # repeat 4
        x = self.choice_block[4][random[4]](x)
        x = self.choice_block[5][random[5]](x)
        x = self.choice_block[6][random[6]](x)
        x = self.choice_block[7][random[7]](x)

        # repeat 8
        x = self.choice_block[8][random[8]](x)
        x = self.choice_block[9][random[9]](x)
        x = self.choice_block[10][random[10]](x)
        x = self.choice_block[11][random[11]](x)
        x = self.choice_block[12][random[12]](x)
        x = self.choice_block[13][random[13]](x)
        x = self.choice_block[14][random[14]](x)
        x = self.choice_block[15][random[15]](x)

        # repeat 4
        x = self.choice_block[16][random[16]](x)
        x = self.choice_block[17][random[17]](x)
        x = self.choice_block[18][random[18]](x)
        x = self.choice_block[19][random[19]](x)

        x = self.conv2(x)
        x = self.gap(x)
        x = x.view(-1, 1024)
        x = self.fc(x)

        return x
