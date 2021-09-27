import torch
import time

import torch.nn as nn
import torch.nn.functional as F

from model.net.backbone import Conv_BN_Act


# from mish_cuda import MishCuda as Mish

# class hswish(nn.Module):
#     def forward(self, x):
#         out = x * F.relu6(x + 3, inplace=True) / 6
#         return out

# #if mish_cuda is not avalible try whats below
class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        device = x.device
        x = x * (torch.tanh(F.softplus(x)))
        x.to(device)
        return x


class Inverted_Bottleneck(nn.Module):

    def __init__(self, c1, c2, s=8, k=3, stride=1):
        super(Inverted_Bottleneck, self).__init__()

        sc1 = int(s*c1)

        self.conv1 = nn.Conv2d(c1, sc1, 1, stride)
        self.bn1 = nn.BatchNorm2d(sc1)
        self.act1 = Mish()

        self.conv2 = nn.Conv2d(sc1, sc1, k, padding=1, dilation=1, groups=sc1)
        self.bn2 = nn.BatchNorm2d(sc1)
        self.act2 = Mish()

        self.conv3 = nn.Conv2d(sc1, c2, 1)
        self.bn3 = nn.BatchNorm2d(c2)
        self.act3 = Mish()

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)

        return x


class Fused_IBN(nn.Module):

    def __init__(self, c1, c2, k=3, s=8, stride=1):
        super(Fused_IBN, self).__init__()

        if c1 != c2:
            self.reduce = True
            c1 = c2
        else:
            self.reduce = False

        self.conv = Conv_BN_Act(int(c1*2), c1, 3, 1, 'mish')

        sc1 = int(s*c1)

        self.conv1 = nn.Conv2d(c1, sc1, k, stride, padding=1)
        self.bn1 = nn.BatchNorm2d(sc1)
        self.act1 = Mish()

        self.convh = Conv_BN_Act(c1, c1 // 2, 3, 1, 'mish')

        self.conv2 = nn.Conv2d(sc1, c2, 1)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act2 = Mish()

    def forward(self, x):

        if self.reduce:
            x = self.conv(x)

        short = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = x + short

        return x


class Tucker(nn.Module):

    def __init__(self, c1, c2, k=3, s=0.25, e=0.75, stride=1):
        super(Tucker, self).__init__()

        sc1 = int(s*c1)
        ec2 = int(e*c2)

        if c1 != c2:
            self.reduce = True
            c1 = c2
        else:
            self.reduce = False

        self.conv = Conv_BN_Act(int(c1*2), c1, 3, 1, 'mish')

        self.conv1 = nn.Conv2d(c1, sc1, 1, stride)
        self.bn1 = nn.BatchNorm2d(sc1)
        self.act1 = Mish()

        self.conv2 = nn.Conv2d(sc1, ec2, k, padding=1)
        self.bn2 = nn.BatchNorm2d(ec2)
        self.act2 = Mish()

        self.conv3 = nn.Conv2d(ec2, c2, 1)
        self.bn3 = nn.BatchNorm2d(c2)
        self.act3 = Mish()

    def forward(self, x):

        if self.reduce:
            x = self.conv(x)

        short = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)

        x = x + short

        return x


class CSPMobileDets(nn.Module):
    def __init__(self, freeze=False):
        super(CSPMobileDets, self).__init__()
        self.freeze = freeze

        # First block
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = Mish()
        self.csp1_block = Conv_BN_Act(32, 16, 3, 1, 'mish')
        self.tucker1_1 = Tucker(32, 16)

        self.conv1_1 = Conv_BN_Act(32, 16, 3, 1, 'mish')
        self.conv1_2 = Conv_BN_Act(16, 32, 3, 2, 'mish')

        # Second block
        self.csp2_block = Conv_BN_Act(32, 16, 3, 1, 'mish')
        self.fused2_1 = Fused_IBN(32, 16)
        self.tucker2_2 = Tucker(16, 16, e=0.25)
        self.tucker2_3 = Tucker(16, 16, e=0.25)
        self.tucker2_4 = Tucker(16, 16, e=0.25)

        self.conv2_1 = Conv_BN_Act(32, 32, 3, 1, 'mish')
        self.conv2_2 = Conv_BN_Act(32, 64, 3, 2, 'mish')

        # Third block
        self.csp3_block = Conv_BN_Act(64, 32, 3, 1, 'mish')
        self.fused3_1 = Fused_IBN(64, 32)
        self.fused3_2 = Fused_IBN(32, 32)
        self.fused3_3 = Fused_IBN(32, 32)
        self.fused3_4 = Fused_IBN(32, 32)

        self.fused3_5 = Fused_IBN(32, 32)
        self.fused3_6 = Fused_IBN(32, 32, s=4)
        self.fused3_7 = Fused_IBN(32, 32, s=4)
        self.fused3_8 = Fused_IBN(32, 32, s=4)

        self.conv3_1 = Conv_BN_Act(64, 64, 3, 1, 'mish')
        self.conv3_2 = Conv_BN_Act(64, 128, 3, 2, 'mish')

        # Fourth block
        self.csp4_block = Conv_BN_Act(128, 64, 3, 1, 'mish')
        self.fused4_1 = Fused_IBN(128, 64)
        self.fused4_2 = Fused_IBN(64, 64, s=4)
        self.fused4_3 = Fused_IBN(64, 64, s=4)
        self.fused4_4 = Fused_IBN(64, 64, s=4)

        self.fused4_5 = Fused_IBN(64, 64)
        self.fused4_6 = Fused_IBN(64, 64)
        self.fused4_7 = Fused_IBN(64, 64)
        self.fused4_8 = Fused_IBN(64, 64)

        self.conv4_1 = Conv_BN_Act(128, 128, 3, 1, 'mish')
        self.conv4_2 = Conv_BN_Act(128, 256, 3, 2, 'mish')

        # Fifth block
        self.csp5_block = Conv_BN_Act(256, 128, 3, 1, 'mish')
        self.fused5_1 = Fused_IBN(256, 128, s=4,)
        self.fused5_2 = Fused_IBN(128, 128, s=4,)
        self.fused5_3 = Fused_IBN(128, 128, s=4)
        self.fused5_4 = Fused_IBN(128, 128, s=4)

        self.conv5_e = Conv_BN_Act(256, 256, 3, 1, 'mish')

    def framework(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        csp1 = self.csp1_block(x)
        x = self.tucker1_1(x)

        x = torch.cat([x, csp1], 1)
        x = self.conv1_1(x)
        x = self.conv1_2(x)

        # Second block
        csp2 = self.csp2_block(x)
        x = self.fused2_1(x)
        x = self.tucker2_2(x)
        x = self.tucker2_3(x)
        x = self.tucker2_4(x)

        x = torch.cat([x, csp2], 1)
        x = self.conv2_1(x)
        x = self.conv2_2(x)

        # Third block
        csp3 = self.csp3_block(x)
        x = self.fused3_1(x)
        x = self.fused3_2(x)
        x = self.fused3_3(x)
        x = self.fused3_4(x)

        x = self.fused3_5(x)
        x = self.fused3_6(x)
        x = self.fused3_7(x)
        x = self.fused3_8(x)

        x = torch.cat([x, csp3], 1)
        x = self.conv3_1(x)
        c1 = x
        x = self.conv3_2(x)

        # Fourth block
        csp4 = self.csp4_block(x)
        x = self.fused4_1(x)
        x = self.fused4_2(x)
        x = self.fused4_3(x)
        x = self.fused4_4(x)

        x = self.fused4_5(x)
        x = self.fused4_6(x)
        x = self.fused4_7(x)
        x = self.fused4_8(x)

        x = torch.cat([x, csp4], 1)
        x = self.conv4_1(x)
        c2 = x
        x = self.conv4_2(x)

        # Fifth block
        csp5 = self.csp5_block(x)
        x = self.fused5_1(x)
        x = self.fused5_2(x)
        x = self.fused5_3(x)
        x = self.fused5_4(x)

        x = torch.cat([x, csp5], 1)
        x = self.conv5_e(x)

        c3 = x

        return c3, c2, c1

    def forward(self, x):

        if self.freeze:
            with torch.no_grad():
                c3, c2, c1 = self.framework(x)
        else:
            c3, c2, c1 = self.framework(x)

        return c3, c2, c1


