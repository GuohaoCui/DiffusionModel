#使用sigmoid激活
import math
import random
from typing import Union

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import init

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class stochasticReLU(nn.Module):
    def __init__(self, low, high):
        super().__init__()
        self.low = low
        self.high = high

    def forward(self, x):

        if random.randint(1, 4) < 3:
            x[x <= 0] = 0
            x3 = x
        else:
            x = x.to(device)
            dim0, dim1, dim2, dim3 = x.shape
            x[x <= 0] = 0
            x0 = x
            x1 = x0.reshape(1, -1)
            K = [random.uniform(self.low, self.high) for _ in range(dim0 * dim1 * dim2 * dim3)]
            K_tensor = torch.tensor(K).to(device)
            K1 = K_tensor.reshape(1, -1)
            x2 = x1 * K1
            x3 = x2.reshape(dim0, dim1, dim2, dim3)

        return x3


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.se = nn.Sequential(
        #     nn.Conv2d(channel, channel // reduction, 1, bias=False),
        #     nn.ReLU(),
        #     # nn.stochasticReLU(),
        #     nn.Conv2d(channel // reduction, channel, 1, bias=False)
        # )
        self.sigmoid = nn.Sigmoid()
        self.Conv2_0 = nn.Conv2d(channel, channel // reduction, 1, bias=False)
        self.Conv2_1 = nn.Conv2d(channel // reduction, channel, 1, bias=False)
        self.active = stochasticReLU(low=math.tan(0.52),high=math.tan(1.05))
    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        # print("max_result.shape",max_result.shape) #([16,512,1,1])

        #重写这个方法
        # print("max_results.shape",max_result.shape)
        max_out_1 = self.Conv2_0(max_result)
        # print("max_results_1.shape1",max_out_1.shape)
        max_out_2 = self.active(max_out_1)
        # print("max_out_2.shape",max_out_2.shape)
        max_out = self.Conv2_1(max_out_2)

        # avg_out = self.se(avg_result)
        avg_out1 = self.Conv2_0(avg_result)

        avg_out2 = self.active(avg_out1)

        avg_out = self.Conv2_1(avg_out2)


        output = self.sigmoid(max_out + avg_out)
        return output

#sa
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=8):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding="same")
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        # print(result.shape)  #([36,512,8,8])
        output = self.conv(result) #([36,1,9,9])
        output = self.sigmoid(output)
        # print(output.shape)
        return output


class CBAMBlock(nn.Module):

    def __init__(self, channel=512, reduction=16, kernel_size=8):
        super().__init__()
        # print(channel,reduction,kernel_size)
        self.ca = ChannelAttention(channel=channel, reduction=reduction).cuda()
        self.sa = SpatialAttention(kernel_size=kernel_size).cuda()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual = x
        # print("x.type",x.type)
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out + residual


# if __name__ == '__main__':
#     input = torch.randn(36, 512, 8, 8)
#     kernel_size = input.shape[2]
#     cbam = CBAMBlock(channel=512, reduction=16, kernel_size=kernel_size)
#     output = cbam(input)
#     print(output.shape)