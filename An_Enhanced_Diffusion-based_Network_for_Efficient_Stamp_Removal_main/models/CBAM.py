#使用sigmoid激活
import numpy as np
import torch
from torch import nn
from torch.nn import init

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1).to(device)
        self.avgpool = nn.AdaptiveAvgPool2d(1).to(device)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False).to(device),
            nn.ReLU().to(device),
            nn.Conv2d(channel // reduction, channel, 1, bias=False).to(device)
        )
        self.Conv2 = nn.Conv2d(channel,channel//reduction,1,bias=False).to(device)

        self.sigmoid = nn.Sigmoid().to(device)

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output

#sa
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=8):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding="same").to(device)
        self.sigmoid = nn.Sigmoid().to(device)

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
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

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
        out = x * self.ca(x)

        # print(out.shape)

        # bbb = self.sa(out)

        # print(bbb.shape)

        # print("out.shape",out.shape)
        # print(self.sa(out))
        out = out * self.sa(out)
        return out + residual


# if __name__ == '__main__':
#     input = torch.randn(36, 512, 8, 8)
#     kernel_size = input.shape[2]
#     cbam = CBAMBlock(channel=512, reduction=16, kernel_size=kernel_size)
#     output = cbam(input)
    # print(output.shape)