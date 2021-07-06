import torch
import time

import torch.nn as nn
import torch.nn.functional as F

from timm.models.efficientnet_blocks import *
from tucker_conv.conv import TuckerConv

# class hswish(nn.Module):
#     def forward(self, x):
#         out = x * F.relu6(x + 3, inplace=True) / 6
#         return out

# Target: Jetson Xavier GPU
class MobileDetGPU(nn.Module):
    def __init__(self, act=nn.ReLU6):
        super(MobileDetGPU, self).__init__()
        
        # First block
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1, stride = 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.ReLU6()
        self.tucker1 = TuckerConv(32, 16, residual = False)
        
        # Second block
        self.fused1 = EdgeResidual(16, 32, exp_ratio = 8, stride = 2, act_layer = nn.ReLU6)
        self.tucker2 = TuckerConv(32, 32, out_comp_ratio = 0.25)
        self.tucker3 = TuckerConv(32, 32, out_comp_ratio = 0.25)
        self.tucker4 = TuckerConv(32, 32, out_comp_ratio = 0.25)
        
        # Third block
        self.fused2 = EdgeResidual(32, 64, exp_ratio = 8, stride = 2, act_layer = nn.ReLU6)
        self.fused3 = EdgeResidual(64, 64, exp_ratio = 8, act_layer = nn.ReLU6)
        self.fused4 = EdgeResidual(64, 64, exp_ratio = 8, act_layer = nn.ReLU6)
        self.fused5 = EdgeResidual(64, 64, exp_ratio = 4, act_layer = nn.ReLU6)
        self.ibn1 = InvertedResidual(64, 256, exp_ratio=8)
        
        # Fourth block
        self.fused6 = EdgeResidual(64, 128, exp_ratio = 8, stride = 2, act_layer = nn.ReLU6)
        self.fused7 = EdgeResidual(128, 128, exp_ratio = 4, act_layer = nn.ReLU6)
        self.fused8 = EdgeResidual(128, 128, exp_ratio = 4, act_layer = nn.ReLU6)
        self.fused9 = EdgeResidual(128, 128, exp_ratio = 4, act_layer = nn.ReLU6)

        
        self.fused10 = EdgeResidual(128, 128, exp_ratio = 8, act_layer = nn.ReLU6)
        self.fused11 = EdgeResidual(128, 128, exp_ratio = 8, act_layer = nn.ReLU6)
        self.fused12 = EdgeResidual(128, 128, exp_ratio = 8, act_layer = nn.ReLU6)
        self.fused13 = EdgeResidual(128, 128, exp_ratio = 8, act_layer = nn.ReLU6)
        self.ibn2 = InvertedResidual(128, 512, exp_ratio=8)
        
        # Fifth block
        self.fused14 = EdgeResidual(128, 128, exp_ratio = 4, stride = 2, act_layer = nn.ReLU6)
        self.fused15 = EdgeResidual(128, 128, exp_ratio = 4, act_layer = nn.ReLU6)
        self.fused16 = EdgeResidual(128, 128, exp_ratio = 4, act_layer = nn.ReLU6)
        self.fused17 = EdgeResidual(128, 128, exp_ratio = 4, act_layer = nn.ReLU6)
        self.ibn3 = InvertedResidual(128, 1024, exp_ratio = 8)
    
    def forward(self, x):
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


if __name__ == '__main__':


    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        timeall = 0
        img = torch.rand(size=(1, 3, 416, 416)).to(device)

        model = MobileDetGPU().to(device)
        model.eval()

        c1, c2, c3 = model(img)
        print(c1.size())
        print(c2.size())
        print(c3.size())

        # for i in range(50):
        #     start = time.time()
        #
        #     c1, c2, c3 = model(img)
        #
        #     print(c1.size())
        #     print(c2.size())
        #     print(c3.size())
        #     # print(c4.size())
        #     # print(c5.size())
        #     end = time.time()
        #     if i == 0:
        #         continue
        #     else:
        #         timeall = timeall + end - start
        #
        #     print(end - start)
        #
        # print("avg time: ", timeall * 1000 / 50, " ms")

