#encoding=utf-8

'''
@Time          : 2020/11/17 08:40
@Author        : Inacmor
@File          : body.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :

'''

import torch.nn as nn
import torch.nn.functional as F
import torch
from model.net.backbone import Conv_BN_Act, Darknet53
from model.net.mobiledet_gpu import MobileDetGPU


class SPP(nn.Module):

    def __init__(self, in_channels):
        super(SPP, self).__init__()

        self.spp_head = Conv_BN_Act(in_channels, in_channels // 2,
                                    kernel_size=1, stride=1, activation='leaky')

        # #SPP层
        self.spp_body1 = nn.MaxPool2d(kernel_size=13, stride=1, padding=13 // 2)
        self.spp_body2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=9 // 2)
        self.spp_body3 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)

    def forward(self, x):

        head = self.spp_head(x)
        p1 = self.spp_body1(head)
        p2 = self.spp_body2(head)
        p3 = self.spp_body3(head)

        output = torch.cat([head, p1, p2, p3], 1)

        return output

# @torchsnooper.snoop()
class FiveStr(nn.Module):

    def __init__(self, inchannels):
        super(FiveStr, self).__init__()

        self.five_str = nn.ModuleList([Conv_BN_Act(inchannels, inchannels // 2,
                                                   kernel_size=1, stride=1, activation='leaky'),
                                       Conv_BN_Act(inchannels // 2, inchannels,
                                                   kernel_size=3, stride=1, activation='leaky'),
                                       Conv_BN_Act(inchannels, inchannels // 2,
                                                   kernel_size=1, stride=1, activation='leaky'),
                                       Conv_BN_Act(inchannels // 2, inchannels,
                                                   kernel_size=3, stride=1, activation='leaky'),
                                       Conv_BN_Act(inchannels, inchannels // 2,
                                                   kernel_size=1, stride=1, activation='leaky')])

    def forward(self, x):

        for layer in self.five_str:
            x = layer(x)

        return x

# @torchsnooper.snoop()
class OUTPUT(nn.Module):

    def __init__(self, in_channels, per_anchors, outs):
        super(OUTPUT, self).__init__()

        self.conv1 = Conv_BN_Act(in_channels, in_channels * 2, 3, 1, 'leaky')
        self.conv2 = nn.Conv2d(in_channels * 2, per_anchors * outs, kernel_size=1, stride=1)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)

        return x

# @torchsnooper.snoop()
class UPSAMPLE(nn.Module):

    def __init__(self, in_channels):
        super(UPSAMPLE, self).__init__()

        self.conv1 = Conv_BN_Act(in_channels, in_channels // 2, kernel_size=1, stride=1, activation='leaky')

    def forward(self, x, up_times, inference=False):
        x = self.conv1(x)

        x = F.interpolate(x, size=(x.size(2) * up_times, x.size(3) * up_times), mode='nearest')

        return x

# @torchsnooper.snoop()
class DOWNSAMPLE(nn.Module):

    def __init__(self, in_channels):
        super(DOWNSAMPLE, self).__init__()

        self.down = Conv_BN_Act(in_channels, in_channels * 2, kernel_size=3, stride=2, activation='leaky')

    def forward(self, x):
        x = self.down(x)

        return x

# @torchsnooper.snoop()
class PANUP1(nn.Module):

    def __init__(self, in_channels):
        super(PANUP1, self).__init__()

        self.conv1 = Conv_BN_Act(in_channels, in_channels // 2, 1, 1, 'leaky')

        self.conv2 = Conv_BN_Act(in_channels // 2, in_channels, 3, 1, 'leaky')

        self.spp = SPP(in_channels)

        self.conv3 = Conv_BN_Act(in_channels * 2, in_channels // 2, 1, 1, 'leaky')

        self.conv4 = Conv_BN_Act(in_channels // 2, in_channels, 3, 1, 'leaky')

        self.conv5 = Conv_BN_Act(in_channels, in_channels // 2, 1, 1, 'leaky')

        self.upsample = UPSAMPLE(in_channels // 2)

    def forward(self, x, uptimes):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.spp(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        up_input = x
        up_output = self.upsample(up_input, uptimes)

        return x, up_output

# @torchsnooper.snoop()
class PANUP2(nn.Module):

    def __init__(self, in_channels):
        super(PANUP2, self).__init__()

        self.conv1 = Conv_BN_Act(in_channels, in_channels // 2, 1, 1, 'leaky')

        self.five_s = FiveStr(in_channels)

        self.upsample = UPSAMPLE(in_channels // 2)

    def forward(self, x, up1, uptimes):

        x = self.conv1(x)

        x = torch.cat([x, up1], 1)

        x = self.five_s(x)

        up_input = x
        up_output = self.upsample(up_input, uptimes)

        return x, up_output

# @torchsnooper.snoop()
class PANDOWN1(nn.Module):

    def __init__(self, in_channels):
        super(PANDOWN1, self).__init__()

        self.conv1 = Conv_BN_Act(in_channels, in_channels // 2, 1, 1, 'leaky')
        self.conv2 = FiveStr(in_channels)
        self.down1 = DOWNSAMPLE(in_channels // 2)

    def forward(self, x, up2):

        x = self.conv1(x)

        x = torch.cat([x, up2], 1)
        x = self.conv2(x)

        down1_input = x
        down1_input = self.down1(down1_input)

        return x, down1_input

# @torchsnooper.snoop()
class PANDOWN2(nn.Module):

    def __init__(self, in_channels):
        super(PANDOWN2, self).__init__()

        self.five_s = FiveStr(in_channels)

        self.down2 = DOWNSAMPLE(in_channels // 2)

    def forward(self, x, down1_output):

        x = torch.cat([x, down1_output], 1)
        x = self.five_s(x)

        down2_input = x

        down2_output = self.down2(down2_input)

        return x, down2_output


# @torchsnooper.snoop()
class YOLOBODY(nn.Module):

    def __init__(self, in_channels, anchors, num_bbparas, num_classes, freeze=False):
        super(YOLOBODY, self).__init__()

        self.backbone = MobileDetGPU(freeze=freeze)
        self.panup1 = PANUP1(in_channels)
        self.five_d32 = FiveStr(in_channels)
        self.outs1 = OUTPUT(in_channels // 2,
                            per_anchors=anchors // 3,
                            outs=num_bbparas + 1 + num_classes)

        self.panup2 = PANUP2(in_channels // 2)
        self.pandown1 = PANDOWN1(in_channels // 4)
        self.outs2 = OUTPUT(in_channels // 4,
                            per_anchors=anchors // 3,
                            outs=num_bbparas + 1 + num_classes)

        self.pandown2 = PANDOWN2(in_channels // 2)
        self.outs3 = OUTPUT(in_channels // 8,
                            per_anchors=anchors // 3,
                            outs=num_bbparas + 1 + num_classes)

    def forward(self, inputs, uptimes1=2, uptimes2=2):

        darknet_out1, darknet_out2, darknet_out3 = self.backbone(inputs)

        d32, up1_output = self.panup1(darknet_out1, uptimes1)

        d16, up2_output = self.panup2(darknet_out2, up1_output, uptimes2)

        d08, down1_output = self.pandown1(darknet_out3, up2_output)

        d16, down2_output = self.pandown2(d16, down1_output)

        d32 = torch.cat([d32, down2_output], 1)
        d32 = self.five_d32(d32)

        feats01 = self.outs1(d32)
        feats02 = self.outs2(d16)
        feats03 = self.outs3(d08)

        return [feats01, feats02, feats03]

















