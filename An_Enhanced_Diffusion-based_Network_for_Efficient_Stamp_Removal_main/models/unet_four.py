r"""
20230313加入了DAC和RMP，其中改进了DAC的结构，在3x3卷积层之后加入了一层1x1卷积层
20230314在DAC前加入了ExternalAttention，并改变了ExternalAttention的结构
20230318使用自注意力机制 参数：学习率调大1/4, epoch 3000, batchsize 1
20230319去掉自学习参数
20230320使用CBAM, 去掉自注意力
20230321改了CBAM里的激活函数
20230321_2卷积模块换CBAM
20230323设计了新模块，将CBAM和多通道注意力融合
20230327取消L1正则化（损失函数），使用Batchnorm
"""
import random

import torch
from torch import nn,optim
import torch.nn.functional as F
import math
from torch.nn import init

import models.four2 as CBAM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_timestep_embedding(timesteps, embedding_dim):
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=16, num_channels=in_channels, eps=1e-6, affine=True)

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(
            x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.temb_proj = torch.nn.Linear(temb_channels,
                                         out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h

class DACblock(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1).to(device)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3).to(device)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5).to(device)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0).to(device)
        # self.para1 = nn.Parameter(torch.normal(0,1, size=(1,1)))
        # self.para2 = nn.Parameter(torch.normal(0,1,size=(1,1)))
        # self.Attnblock = AttnBlock(channel)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        #  x   ([32,512,8,8])
        # src1 = x
        #短跳连  自学习参数1
        # Softmax = nn.Softmax(dim=0)
        # para1 = Softmax(self.para1)
        # para1 = nn.functional.normalize(para1)
        # cbam = CBAM.CBAMBlock(channel=x.shape[1],reduction=16,kernel_size=x.shape[2])
        # src2 = self.Attnblock(src1) + x  #([32,64,512])
        # src2 = cbam(src1) + x
        #长跳连  自学习参数2
        # para2 = Softmax(self.para2)
        # para2 = nn.functional.normalize(para2)
        # src3 = src2 + x

        dilate1_out = nonlinearity(self.conv1x1(self.dilate1(x)))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate1(self.conv1x1(self.dilate2(x)))))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate3(self.conv1x1(self.dilate2(self.conv1x1(self.dilate1(x)))))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate1(x))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out

#RMPblock
class RMPblock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2).to(device)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=3).to(device)
        self.pool3 = nn.MaxPool2d(kernel_size=(5, 5), stride=5).to(device)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6).to(device)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0).to(device)
        self.conv2 = nn.Conv2d(in_channels=516, out_channels=512, kernel_size=1, padding=0).to(device)
    def forward(self, x):

        # src1 = x #([16,512,64])
        #
        # cbam = CBAM.CBAMBlock(channel=x.shape[1],reduction=16,kernel_size=x.shape[2])
        # src2 = cbam(src1) + x

        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
        self.layer1 = F.interpolate(self.conv(self.pool1(x)), size=(h, w), mode='bilinear', align_corners=True)
        self.layer2 = F.interpolate(self.conv(self.pool2(x)), size=(h, w), mode='bilinear', align_corners=True)
        self.layer3 = F.interpolate(self.conv(self.pool3(x)), size=(h, w), mode='bilinear', align_corners=True)
        self.layer4 = F.interpolate(self.conv(self.pool4(x)), size=(h, w), mode='bilinear', align_corners=True)
        out = torch.cat([self.layer1, self.layer2, self.layer3,self.layer4 ,x], 1)#([16,516,8,8])
        # out = out.view(16,512,8,8)
        # print("out.shape",out.shape)
        out = self.conv2(out)
        # print("out.shape",out.shape)
        return out

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        # print("h_.shape000",h_.shape) #([32,384,16,16])
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # 自注意力
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h*w)  # b,c,hw
        w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        # print("x+h_.shape111",h_.shape)#([32,384,16,16])

        return x+h_

#class stochasticReLU(nn.Module):
#    def __init__(self, low, high):
#        super().__init__()
#        self.low = low
#        self.high = high

#    def forward(self, x):

#        if random.randint(1, 4) < 3:
#            x[x <= 0] = 0
#            x3 = x
#        else:
#            x = x.to(device)
#            dim0, dim1, dim2, dim3 = x.shape
#            x[x <= 0] = 0
#            x0 = x
#            x1 = x0.reshape(1, -1)
#            K = [random.uniform(self.low, self.high) for _ in range(dim0 * dim1 * dim2 * dim3)]
#            K_tensor = torch.tensor(K).to(device)
#            K1 = K_tensor.reshape(1, -1)
#            x2 = x1 * K1
#            x3 = x2.reshape(dim0, dim1, dim2, dim3)

#        return x3


class stochasticReLU(nn.Module):
    def __init__(self, low, high):
        super().__init__()
        self.low = low
        self.high = high
    def forward(self, x):
        device = x.device
        if random.randint(1, 4) < 3:
            x[x <= 0] = 0
            x3 = x
        else:
            dim0, dim1, dim2, dim3 = x.shape
            x[x <= 0] = 0
            x0 = x
            x1 = x0.reshape(1, -1)
            K_tensor = torch.rand(dim0 * dim1 * dim2 * dim3, device=device) * (self.high - self.low) + self.low
            K1 = K_tensor.reshape(1, -1)
            x2 = x1 * K1
            x3 = x2.reshape(dim0, dim1,dim2,dim3)
        return x3





class DiffusionUNet(nn.Module):
    def __init__(self,config):
        super().__init__()

        # ch, out_ch, ch_mult = 128,3,tuple([1, 2, 3, 4])
        # num_res_blocks = 2
        # attn_resolutions = [16, ]
        # dropout = 0.0
        # in_channels = 3 * 2 if True else 3
        # resolution = 64
        # resamp_with_conv = True

        self.config = config
        ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(config.model.ch_mult)
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        in_channels = config.model.in_channels * 2 if config.data.conditional else config.model.in_channels
        resolution = config.data.image_size
        resamp_with_conv = config.model.resamp_with_conv


        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch,
                            self.temb_ch),
            torch.nn.Linear(self.temb_ch,
                            self.temb_ch),
        ])

        # 下采样
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+ch_mult
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        self.active = stochasticReLU(low=0.57735,high=1.73205)


        # middle
        self.mid = nn.Module()

        # self.mid.block_00 = ExternalAttention(512,64)

        # self.mid.block_0 = DACblock(channel=block_in)

        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        # print('self.mid.attn_1',self.mid.attn_1)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        # self.mid.block_3=RMPblock(in_channels=block_in)

        # 上采样
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x, t):
        # print("x.shape",x.shape)
        # print(self.resolution)
        assert x.shape[2] == x.shape[3] == self.resolution

        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        # 下采样
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1] #(16,512,8,8)

        # h = self.mid.block_0(h)
        # print("h.shape1",h.shape)

        # print("h.shape",h.shape) #([16,512,8,8])
#新模块

        batchnorm = nn.BatchNorm2d(h.shape[1],eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None).to(device)

        h_batchnorm = batchnorm(h)

        cbam = CBAM.CBAM(h.shape[1])

        # print("h_batchnorm.shape",h_batchnorm.shape)
        h_cbam = cbam(h_batchnorm)
        # print("h_cbam",h_cbam.shape)
        h_cbamAndH = h_cbam + h

        # ReLU = nn.ReLU()
        h_ReLU = self.active(h_cbamAndH)

        # print("h.shape",h.shape)  #([16,512,8,8])
        dac = DACblock(channel=h.shape[1])
        h_dac = dac(h)
        # print("h_dac.shape",h_dac.shape) #([16,512,8,8])
        h_cbamAndDac = torch.cat([h_ReLU, h_dac], 1)
        # print("h_cbamAndDac",h_cbamAndDac.shape) #([16,1024,8,8])
        convlow = nn.Conv2d(h_cbamAndDac.shape[1],h.shape[1],1,1).to(device)
        h_convlowAndH = convlow(h_cbamAndDac) + h
        # print("h_convlow",h_convlow.shape)


        h_ReLU2 = self.active(h_convlowAndH)
        # print("h_ReLU2.shape",h_ReLU2.shape)


        h_out = h_ReLU2 + h


        h = self.mid.block_1(h_out, temb)
        # print("h.shape2",h.shape)
        h = self.mid.attn_1(h)
        # print("h.shape3",h.shape)
        h = self.mid.block_2(h, temb)
        # print("h.shape4",h.shape)
        # h = self.mid.block_3(h)
        # print("h.shape5",h.shape)

# 池化模块

        batchnorm1 = nn.BatchNorm2d(h.shape[1],eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None).to(device)

        h_batchnorm1 = batchnorm1(h)

        # cbam = CBAM.CBAM()
        h_cbam = cbam(h_batchnorm1)

        # print(type(h_cbam))

        h_cbamAndH = h_cbam + h
        ReLU = nn.ReLU()
        h_ReLU = ReLU(h_cbamAndH)
        # print("h.shape",h.shape)  #([16,512,8,8])
        # dac = DACblock(channel=h.shape[1])
        rmp = RMPblock(in_channels=h.shape[1])
        h_rmp = rmp(h)
        # print("h_dac.shape",h_dac.shape) #([16,512,8,8])
        h_cbamAndDac = torch.cat([h_ReLU, h_rmp], 1)
        # print("h_cbamAndDac",h_cbamAndDac.shape) #([16,1024,8,8])
        convlow = nn.Conv2d(h_cbamAndDac.shape[1],h.shape[1],1,1).to(device)
        h_convlowAndH = convlow(h_cbamAndDac) + h
        # print("h_convlow",h_convlow.shape)
        h_ReLU2 = ReLU(h_convlowAndH)
        # print("h_ReLU2.shape",h_ReLU2.shape)
        h_out = h_ReLU2 + h
        h = h_out


        # 上采样
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

# net = DiffusionUNet()
# print(net)
