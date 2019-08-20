#choice_block with stride = 1
from choice_block import Choice_Block3_1, Choice_Block5_1, Choice_Block7_1, Choice_Blockx_1
#choice_block with stride = 2
from choice_block import Choice_Block3_2, Choice_Block5_2, Choice_Block7_2, Choice_Blockx_2

import torch.nn as nn


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)

        # **********************************repeat4*************************************************#
        # choice_block0 stride=2
        self.choice_3_0 = Choice_Block3_2(16, 64//2)
        self.choice_5_0 = Choice_Block5_2(16, 64//2)
        self.choice_7_0 = Choice_Block7_2(16, 64//2)
        self.choice_x_0 = Choice_Blockx_2(16, 64//2)
        self.choice_block0 = [self.choice_3_0, self.choice_5_0, self.choice_7_0, self.choice_x_0]

        # choice_block1 stride=1
        self.choice_3_1 = Choice_Block3_1(64//2, 64//2)
        self.choice_5_1 = Choice_Block5_1(64//2, 64//2)
        self.choice_7_1 = Choice_Block7_1(64//2, 64//2)
        self.choice_x_1 = Choice_Blockx_1(64//2, 64//2)
        self.choice_block1 = [self.choice_3_1, self.choice_5_1, self.choice_7_1, self.choice_x_1]

        # choice_block2 stride=1
        self.choice_3_2 = Choice_Block3_1(64//2, 64//2)
        self.choice_5_2 = Choice_Block5_1(64//2, 64//2)
        self.choice_7_2 = Choice_Block7_1(64//2, 64//2)
        self.choice_x_2 = Choice_Blockx_1(64//2, 64//2)
        self.choice_block2 = [self.choice_3_2, self.choice_5_2, self.choice_7_2, self.choice_x_2]

        # choice_block3 stride=1
        self.choice_3_3 = Choice_Block3_1(64//2, 64//2)
        self.choice_5_3 = Choice_Block5_1(64//2, 64//2)
        self.choice_7_3 = Choice_Block7_1(64//2, 64//2)
        self.choice_x_3 = Choice_Blockx_1(64//2, 64//2)
        self.choice_block3 = [self.choice_3_3, self.choice_5_3, self.choice_7_3, self.choice_x_3]

        # ***********************************repeat4************************************************#
        # choice_block4 stride=2
        self.choice_3_4 = Choice_Block3_2(64, 160//2)
        self.choice_5_4 = Choice_Block5_2(64, 160//2)
        self.choice_7_4 = Choice_Block7_2(64, 160//2)
        self.choice_x_4 = Choice_Blockx_2(64, 160//2)
        self.choice_block4 = [self.choice_3_4, self.choice_5_4, self.choice_7_4, self.choice_x_4]


        # choice_block5 stride=1
        self.choice_3_5 = Choice_Block3_1(160//2, 160//2)
        self.choice_5_5 = Choice_Block5_1(160//2, 160//2)
        self.choice_7_5 = Choice_Block7_1(160//2, 160//2)
        self.choice_x_5 = Choice_Blockx_1(160//2, 160//2)
        self.choice_block5 = [self.choice_3_5, self.choice_5_5, self.choice_7_5, self.choice_x_5]

        # choice_block6 stride=1
        self.choice_3_6 = Choice_Block3_1(160//2, 160//2)
        self.choice_5_6 = Choice_Block5_1(160//2, 160//2)
        self.choice_7_6 = Choice_Block7_1(160//2, 160//2)
        self.choice_x_6 = Choice_Blockx_1(160//2, 160//2)
        self.choice_block6 = [self.choice_3_6, self.choice_5_6, self.choice_7_6, self.choice_x_6]

        # choice_block7 stride=1
        self.choice_3_7 = Choice_Block3_1(160//2, 160//2)
        self.choice_5_7 = Choice_Block5_1(160//2, 160//2)
        self.choice_7_7 = Choice_Block7_1(160//2, 160//2)
        self.choice_x_7 = Choice_Blockx_1(160//2, 160//2)
        self.choice_block7 = [self.choice_3_7, self.choice_5_7, self.choice_7_7, self.choice_x_7]

        # ***********************************repeat8************************************************#
        # choice_block8 stride=2
        self.choice_3_8 = Choice_Block3_2(160, 320//2)
        self.choice_5_8 = Choice_Block5_2(160, 320//2)
        self.choice_7_8 = Choice_Block7_2(160, 320//2)
        self.choice_x_8 = Choice_Blockx_2(160, 320//2)
        self.choice_block8 = [self.choice_3_8, self.choice_5_8, self.choice_7_8, self.choice_x_8]

        # choice_block9 stride=1
        self.choice_3_9 = Choice_Block3_1(320//2, 320//2)
        self.choice_5_9 = Choice_Block5_1(320//2, 320//2)
        self.choice_7_9 = Choice_Block7_1(320//2, 320//2)
        self.choice_x_9 = Choice_Blockx_1(320//2, 320//2)
        self.choice_block9 = [self.choice_3_9, self.choice_5_9, self.choice_7_9, self.choice_x_9]

        # choice_block10 stride=1
        self.choice_3_10 = Choice_Block3_1(320//2, 320//2)
        self.choice_5_10 = Choice_Block5_1(320//2, 320//2)
        self.choice_7_10 = Choice_Block7_1(320//2, 320//2)
        self.choice_x_10 = Choice_Blockx_1(320//2, 320//2)
        self.choice_block10 = [self.choice_3_10, self.choice_5_10, self.choice_7_10, self.choice_x_10]

        # choice_block11 stride=1
        self.choice_3_11 = Choice_Block3_1(320//2, 320//2)
        self.choice_5_11 = Choice_Block5_1(320//2, 320//2)
        self.choice_7_11 = Choice_Block7_1(320//2, 320//2)
        self.choice_x_11 = Choice_Blockx_1(320//2, 320//2)
        self.choice_block11 = [self.choice_3_11, self.choice_5_11, self.choice_7_11, self.choice_x_11]

        # choice_block12 stride=1
        self.choice_3_12 = Choice_Block3_1(320//2, 320//2)
        self.choice_5_12 = Choice_Block5_1(320//2, 320//2)
        self.choice_7_12 = Choice_Block7_1(320//2, 320//2)
        self.choice_x_12 = Choice_Blockx_1(320//2, 320//2)
        self.choice_block12 = [self.choice_3_12, self.choice_5_12, self.choice_7_12, self.choice_x_12]

        # choice_block13 stride=1
        self.choice_3_13 = Choice_Block3_1(320//2, 320//2)
        self.choice_5_13 = Choice_Block5_1(320//2, 320//2)
        self.choice_7_13 = Choice_Block7_1(320//2, 320//2)
        self.choice_x_13 = Choice_Blockx_1(320//2, 320//2)
        self.choice_block13 = [self.choice_3_13, self.choice_5_13, self.choice_7_13, self.choice_x_13]

        # choice_block14 stride=1
        self.choice_3_14 = Choice_Block3_1(320//2, 320//2)
        self.choice_5_14 = Choice_Block5_1(320//2, 320//2)
        self.choice_7_14 = Choice_Block7_1(320//2, 320//2)
        self.choice_x_14 = Choice_Blockx_1(320//2, 320//2)
        self.choice_block14 = [self.choice_3_14, self.choice_5_14, self.choice_7_14, self.choice_x_14]

        # choice_block15 stride=1
        self.choice_3_15 = Choice_Block3_1(320//2, 320//2)
        self.choice_5_15 = Choice_Block5_1(320//2, 320//2)
        self.choice_7_15 = Choice_Block7_1(320//2, 320//2)
        self.choice_x_15 = Choice_Blockx_1(320//2, 320//2)
        self.choice_block15 = [self.choice_3_15, self.choice_5_15, self.choice_7_15, self.choice_x_15]

        # ***********************************repeat4************************************************#
        # choice_block16 stride=2
        self.choice_3_16 = Choice_Block3_2(320, 640//2)
        self.choice_5_16 = Choice_Block5_2(320, 640//2)
        self.choice_7_16 = Choice_Block7_2(320, 640//2)
        self.choice_x_16 = Choice_Blockx_2(320, 640//2)
        self.choice_block16 = [self.choice_3_16, self.choice_5_16, self.choice_7_16, self.choice_x_16]

        # choice_block17 stride=1
        self.choice_3_17 = Choice_Block3_1(640//2, 640//2)
        self.choice_5_17 = Choice_Block5_1(640//2, 640//2)
        self.choice_7_17 = Choice_Block7_1(640//2, 640//2)
        self.choice_x_17 = Choice_Blockx_1(640//2, 640//2)
        self.choice_block17 = [self.choice_3_17, self.choice_5_17, self.choice_7_17, self.choice_x_17]

        # choice_block18 stride=1
        self.choice_3_18 = Choice_Block3_1(640//2, 640//2)
        self.choice_5_18 = Choice_Block5_1(640//2, 640//2)
        self.choice_7_18 = Choice_Block7_1(640//2, 640//2)
        self.choice_x_18 = Choice_Blockx_1(640//2, 640//2)
        self.choice_block18 = [self.choice_3_18, self.choice_5_18, self.choice_7_18, self.choice_x_18]

        # choice_block19 stride=1
        self.choice_3_19 = Choice_Block3_1(640//2, 640//2)
        self.choice_5_19 = Choice_Block5_1(640//2, 640//2)
        self.choice_7_19 = Choice_Block7_1(640//2, 640//2)
        self.choice_x_19 = Choice_Blockx_1(640//2, 640//2)
        self.choice_block19 = [self.choice_3_19, self.choice_5_19, self.choice_7_19, self.choice_x_19]

        self.conv2 = nn.Conv2d(640, 1024, kernel_size=1, stride=1, padding=0)
        self.gap = nn.MaxPool2d(kernel_size=7, stride=1, padding=0)
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x, random):

        x = self.conv1(x)

        # repeat 4
        x = self.choice_block0[random[0]](x)
        x = self.choice_block1[random[1]](x)
        x = self.choice_block2[random[2]](x)
        x = self.choice_block3[random[3]](x)

        # repeat 4
        x = self.choice_block4[random[4]](x)
        x = self.choice_block5[random[5]](x)
        x = self.choice_block6[random[6]](x)
        x = self.choice_block7[random[7]](x)

        # repeat 8
        x = self.choice_block8[random[8]](x)
        x = self.choice_block9[random[9]](x)
        x = self.choice_block10[random[10]](x)
        x = self.choice_block11[random[11]](x)
        x = self.choice_block12[random[12]](x)
        x = self.choice_block13[random[13]](x)
        x = self.choice_block14[random[14]](x)
        x = self.choice_block15[random[15]](x)

        # repeat 4
        x = self.choice_block16[random[16]](x)
        x = self.choice_block17[random[17]](x)
        x = self.choice_block18[random[18]](x)
        x = self.choice_block19[random[19]](x)

        x = self.conv2(x)
        x = self.gap(x)
        x = x.view(-1, 1024)
        x = self.fc(x)

        return x
