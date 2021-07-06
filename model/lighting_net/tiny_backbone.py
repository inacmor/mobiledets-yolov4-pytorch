import torch.nn as nn
import torch.nn.functional as F
import torch
import sys
import math
import numpy as np
import tensorwatch as tw
import torchsnooper


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        device = x.device
        x = x * (torch.tanh(F.softplus(x)))
        x.to(device)
        return x


class Conv_BN_Act(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, activation, bn=True, bias=True):
        super(Conv_BN_Act, self).__init__()

        pad = (kernel_size - 1) // 2

        # #输入conv
        if bias:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False)

        # #输入bn
        if bn:
            self.bn = nn.BatchNorm2d(out_channels)

        # #输入activation
        if activation == 'mish':
            self.activation = Mish()
        elif activation == 'leaky':
            self.activation = nn.LeakyReLU(0.1, inplace=True)# #>0时为y=x，故输出输入只需要一个变量来存储
        elif activation == 'linear':
            pass

        # #报错机制
        else:
            print("activate error !!! {} {} {}".format(sys._getframe().f_code.co_filename,
                                                       sys._getframe().f_code.co_name, sys._getframe().f_lineno))

    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)

        return x

# @torchsnooper.snoop()
class ResN(nn.Module):

    def __init__(self, in_channels, number_blocks):

        super(ResN, self).__init__()

        self.res_n = nn.ModuleList()

        if number_blocks == 1:
            self.res_head = Conv_BN_Act(in_channels, in_channels, 1, 1, 'mish')

            self.res_n.append(Conv_BN_Act(in_channels, in_channels // 2, 1, 1, 'mish'))
            self.res_n.append(Conv_BN_Act(in_channels // 2, in_channels, 3, 1, 'mish'))

            self.res_n_end = (Conv_BN_Act(in_channels, in_channels, 1, 1, 'mish'))
        else:
            self.res_head = Conv_BN_Act(in_channels, in_channels // 2, 1, 1, 'mish')
            for i in range(number_blocks):
                self.res_n.append(Conv_BN_Act(in_channels // 2, in_channels // 2, 1, 1, 'mish'))
                self.res_n.append(Conv_BN_Act(in_channels // 2, in_channels // 2, 3, 1, 'mish'))

            self.res_n_end = Conv_BN_Act(in_channels // 2, in_channels // 2, 1, 1, 'mish')

    def forward(self, x):

        x = self.res_head(x)
        short = x

        for l in self.res_n:

            x = l(x)

        x = x + short

        x = self.res_n_end(x)

        return x

# @torchsnooper.snoop()
class CSPN(nn.Module):

    def __init__(self, in_channels, number_blocks):
        super(CSPN, self).__init__()

        # #添加头部————降维
        self.csp_head = Conv_BN_Act(in_channels, in_channels * 2,
                                    kernel_size=3, stride=2, activation='mish')

        # #添加csp主路径
        self.csp_body = ResN(in_channels * 2, number_blocks)

        # #添加CSP捷径
        if number_blocks == 1:
            self.csp_shortcut = Conv_BN_Act(in_channels * 2, in_channels * 2,
                                            kernel_size=1, stride=1, activation='mish')
        else:
            self.csp_shortcut = Conv_BN_Act(in_channels * 2, in_channels,
                                            kernel_size=1, stride=1, activation='mish')

    def forward(self, x):

        x = self.csp_head(x)
        short = x

        # #主路径
        main = self.csp_body(x)

        # #捷径
        short = self.csp_shortcut(short)

        output = torch.cat([main, short], 1)

        return output


class Darknet53(nn.Module):

    def __init__(self, in_channels=3, top_channels=32, freeze=False):
        super(Darknet53, self).__init__()

        # #冻结层
        self.freeze = freeze

        self.darknent53 = nn.ModuleList()

        self.conv1 = Conv_BN_Act(in_channels, top_channels, 3, 1, 'mish')

        self.csp1 = CSPN(top_channels, number_blocks=1)

        self.conv2 = Conv_BN_Act(top_channels * 4, top_channels * 2, 1, 1, 'mish')

        self.csp2 = CSPN(top_channels * 2, number_blocks=2)

        self.conv3 = Conv_BN_Act(top_channels * 4, top_channels * 4, 1, 1, 'mish')

        self.csp3 = CSPN(top_channels * 4, number_blocks=8)

        # #branch01
        self.conv4 = Conv_BN_Act(top_channels * 8, top_channels * 8, 1, 1, 'mish')

        self.csp4 = CSPN(top_channels * 8, number_blocks=8)

        # #branch02
        self.conv5 = Conv_BN_Act(top_channels * 16, top_channels * 16, 1, 1, 'mish')

        self.csp5 = CSPN(top_channels * 16, number_blocks=4)

        self.conv6 = Conv_BN_Act(top_channels * 32, top_channels * 32, 1, 1, 'mish')

    def forward(self, x):

        if self.freeze:
            with torch.no_grad():
                x = self.conv1(x)
                x = self.csp1(x)

                x = self.conv2(x)
                x = self.csp2(x)

                x = self.conv3(x)
                x = self.csp3(x)

                x = self.conv4(x)
                branch_02 = x
                x = self.csp4(x)

                x = self.conv5(x)
                branch_01 = x
                x = self.csp5(x)

                x = self.conv6(x)

        else:
            x = self.conv1(x)
            x = self.csp1(x)

            x = self.conv2(x)
            x = self.csp2(x)

            x = self.conv3(x)
            x = self.csp3(x)

            x = self.conv4(x)
            x = self.csp4(x)

            x = self.conv5(x)
            branch_01 = x
            x = self.csp5(x)

            x = self.conv6(x)

        return x, branch_01
