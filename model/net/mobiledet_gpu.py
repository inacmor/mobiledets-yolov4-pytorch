import torch
import time

import torch.nn as nn
import torch.nn.functional as F


from mish_cuda import MishCuda as Mish

# class hswish(nn.Module):
#     def forward(self, x):
#         out = x * F.relu6(x + 3, inplace=True) / 6
#         return out


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

        sc1 = int(s*c1)

        self.conv1 = nn.Conv2d(c1, sc1, k, stride, padding=1)
        self.bn1 = nn.BatchNorm2d(sc1)
        self.act1 = Mish()

        self.conv2 = nn.Conv2d(sc1, c2, 1)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act2 = Mish()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        return x


class Tucker(nn.Module):

    def __init__(self, c1, c2, k=3, s=0.25, e=0.75, stride=1):
        super(Tucker, self).__init__()

        sc1 = int(s*c1)
        ec2 = int(e*c2)

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


class MobileDetGPU(nn.Module):
    def __init__(self, freeze=False):
        super(MobileDetGPU, self).__init__()
        self.freeze = freeze

        # First block
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = Mish()
        self.tucker1 = Tucker(32, 16)

        # Second block
        self.fused1 = Fused_IBN(16, 32, stride=2)
        self.tucker2 = Tucker(32, 32, e=0.25)
        self.tucker3 = Tucker(32, 32, e=0.25)
        self.tucker4 = Tucker(32, 32, e=0.25)

        # Third block
        self.fused2 = Fused_IBN(32, 64, stride=2)
        self.fused3 = Fused_IBN(64, 64)
        self.fused4 = Fused_IBN(64, 64)
        self.fused5 = Fused_IBN(64, 64, s=4)
        self.ibn1 = Inverted_Bottleneck(64, 256, s=8)

        # Fourth block
        self.fused6 = Fused_IBN(64, 128, stride=2)
        self.fused7 = Fused_IBN(128, 128, s=4)
        self.fused8 = Fused_IBN(128, 128, s=4)
        self.fused9 = Fused_IBN(128, 128, s=4)

        self.fused10 = Fused_IBN(128, 128)
        self.fused11 = Fused_IBN(128, 128)
        self.fused12 = Fused_IBN(128, 128)
        self.fused13 = Fused_IBN(128, 128)
        self.ibn2 = Inverted_Bottleneck(128, 512, s=8)

        # Fifth block
        self.fused14 = Fused_IBN(128, 128, s=4, stride=2)
        self.fused15 = Fused_IBN(128, 128, s=4)
        self.fused16 = Fused_IBN(128, 128, s=4)
        self.fused17 = Fused_IBN(128, 128, s=4)
        self.ibn3 = Inverted_Bottleneck(128, 1024, s=8)

    def forward(self, x):

        if self.freeze:
            with torch.no_grad():
                # First block
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.act1(x)
                x = self.tucker1(x)

                # Second block
                x = self.fused1(x)
                x = self.tucker2(x)
                x = self.tucker3(x)
                x = self.tucker4(x)

                # Third block
                x = self.fused2(x)
                x = self.fused3(x)
                x = self.fused4(x)
                x = self.fused5(x)
                c1 = x
                c1 = self.ibn1(c1)

                # Fourth block
                x = self.fused6(x)
                x = self.fused7(x)
                x = self.fused8(x)
                x = self.fused9(x)

                x = self.fused10(x)
                x = self.fused11(x)
                x = self.fused12(x)
                x = self.fused13(x)
                c2 = x
                c2 = self.ibn2(c2)

                # Fifth block
                x = self.fused14(x)
                x = self.fused15(x)
                x = self.fused16(x)
                x = self.fused17(x)
                c3 = x
                c3 = self.ibn3(c3)
        else:
            # First block
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.act1(x)
            x = self.tucker1(x)

            # Second block
            x = self.fused1(x)
            x = self.tucker2(x)
            x = self.tucker3(x)
            x = self.tucker4(x)

            # Third block
            x = self.fused2(x)
            x = self.fused3(x)
            x = self.fused4(x)
            x = self.fused5(x)
            c1 = x
            c1 = self.ibn1(c1)

            # Fourth block
            x = self.fused6(x)
            x = self.fused7(x)
            x = self.fused8(x)
            x = self.fused9(x)

            x = self.fused10(x)
            x = self.fused11(x)
            x = self.fused12(x)
            x = self.fused13(x)
            c2 = x
            c2 = self.ibn2(c2)

            # Fifth block
            x = self.fused14(x)
            x = self.fused15(x)
            x = self.fused16(x)
            x = self.fused17(x)
            c3 = x
            c3 = self.ibn3(c3)

        return c3, c2, c1


