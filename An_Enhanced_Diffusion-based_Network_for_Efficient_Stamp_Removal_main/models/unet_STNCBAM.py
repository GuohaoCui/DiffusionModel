r"""
20230313加入了DAC和RMP，其中改进了DAC的结构，在3x3卷积层之后加入了一层1x1卷积层
20230314在DAC前加入了ExternalAttention，并改变了ExternalAttention的结构
20230318使用自注意力机制
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
import models.ODConv2d as ODC
import models.STN_CBAM as CBAM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_timestep_embedding(timesteps, embedding_dim):
    timesteps = timesteps.reshape(-1)
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

# class stochasticReLU(nn.Module):
#     def __init__(self, low, high):
#         super().__init__()
#         self.low = low
#         self.high = high
#
#     def forward(self, x):
#
#         if random.randint(1, 4) < 3:
#             x[x <= 0] = 0
#             x3 = x
#         else:
#             x = x.to(device)
#             dim0, dim1, dim2, dim3 = x.shape
#             x[x <= 0] = 0
#             x0 = x
#             x1 = x0.reshape(1, -1)
#             K = [random.uniform(self.low, self.high) for _ in range(dim0 * dim1 * dim2 * dim3)]
#             K_tensor = torch.tensor(K).to(device)
#             K1 = K_tensor.reshape(1, -1)
#             x2 = x1 * K1
#             x3 = x2.reshape(dim0, dim1, dim2, dim3)
#
#         return x3

# class stochasticReLU(nn.Module):
#     def __init__(self, low, high):
#         super().__init__()
#         self.low = low
#         self.high = high
#
#     def forward(self, x):
#         x = x.to(device)
#         random_number = torch.randint(low=1, high=4, size=(), device='cuda')
#         if random_number < 3:
#             x[x <= 0] = 0
#             x3 = x
#         else:
#             dim0, dim1, dim2, dim3 = x.shape
#             w = torch.randint(low=-5, high=5, size=(), device='cuda')
#             if w==0:
#                x[x <= 0] = 0
#             else:
#                x[x <= w/10] = 0
#             x0 = x
#             x1 = x0.reshape(1, -1)
#             K = torch.empty(dim0, dim1, dim2, dim3, device='cuda')
#             K.uniform_(self.low, self.high)
#             K = K.view(-1)
#             K_tensor = torch.tensor(K).to(device)
#             K1 = K_tensor.reshape(1, -1)
#             if w==0:
#                x2 = x1 * K1
#             else:
#                x2 = x1 * K1 + w/10
#             x3 = x2.reshape(dim0, dim1, dim2, dim3)
#
#         return x3



class stochasticReLU(nn.Module):
    def __init__(self, low, high):
        super().__init__()
        self.low = low
        self.high = high

    def forward(self, x):
        random_number = torch.randint(low=1, high=4, size=(), device='cuda')
        random_mask = (random_number < 3).expand_as(x) # expand the random number to a boolean tensor
        x = torch.where(random_mask, F.relu(x), x) # use torch.where instead of if-else
        dim0, dim1, dim2, dim3 = x.shape
        w = torch.randint(low=-5, high=5, size=(), device='cuda')
        w_mask = (w != 0).expand_as(x) # expand the w to a boolean tensor
        x = torch.where(w_mask, torch.where(x <= w/10, torch.zeros_like(x), x), F.relu(x)) # use torch.where instead of if-else
        x0 = x
        x1 = x0.reshape(1, -1)
        K = torch.empty(dim0, dim1, dim2, dim3, device='cuda')
        K.uniform_(self.low, self.high)
        K = K.view(-1)
        K_tensor = torch.tensor(K).to(device)
        K1 = K_tensor.reshape(1, -1)
        w_tensor = torch.tensor(w).to(device) # convert w to a tensor
        w_tensor = w_tensor.expand_as(K1) # expand w to the same shape as K1
        x2 = torch.where(w_tensor == 0, x1 * K1, x1 * K1 + w/10) # use torch.where instead of if-else
        x3 = x2.reshape(dim0, dim1, dim2, dim3)

        return x3

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


        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
        self.layer1 = F.interpolate(self.conv(self.pool1(x)), size=(h, w), mode='bilinear', align_corners=True)
        self.layer2 = F.interpolate(self.conv(self.pool2(x)), size=(h, w), mode='bilinear', align_corners=True)
        self.layer3 = F.interpolate(self.conv(self.pool3(x)), size=(h, w), mode='bilinear', align_corners=True)
        self.layer4 = F.interpolate(self.conv(self.pool4(x)), size=(h, w), mode='bilinear', align_corners=True)
        out = torch.cat([self.layer1, self.layer2, self.layer3,self.layer4 ,x], 1)#([16,516,8,8])
        out = self.conv2(out)
        return out

from thop import profile

class DiffusionUNet(nn.Module):
    def __init__(self,config):
        super().__init__()

       # ch, out_ch, ch_mult = 64,3,tuple([1, 2, 3, 4])
        #num_res_blocks = 2
        #attn_resolutions = [0, ]
        #dropout = 0.0
        #in_channels = 3 * 2 if True else 3
        #resolution = 16
        #resamp_with_conv = True

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

        #self.active = nn.ReLU(inplace=True)
    def forward(self, x, t):
        print(x.shape[2])
        print(x.shape[3])
        assert x.shape[2] == x.shape[3] == self.resolution
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
#新模块
        batchnorm = nn.BatchNorm2d(h.shape[1],eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None).to(device)
        h_batchnorm = batchnorm(h)
        cbam = CBAM.STNWithCBAM(h.shape[1],h.shape[2]).to(device)
        h_cbam = cbam(h_batchnorm)
        h_cbamAndH = h_cbam + h
        h_ReLU = self.active(h_cbamAndH)
        odc = ODC.ODConv2d(in_planes=h.shape[1],out_planes=h.shape[1],kernel_size=3,padding=1)
        h_odc = odc(h_batchnorm)
        h_ReLU2 = self.active(h_odc)
        h_cbamAndOdc = torch.cat([h_ReLU, h_ReLU2], 1)
        convlow = nn.Conv2d(h_cbamAndOdc.shape[1],h.shape[1],1,1).to(device)
        h_convlowAndH = convlow(h_cbamAndOdc) + h
        h_out = h_convlowAndH


        h = self.mid.block_1(h_out, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)


        h_batchnorm1 = batchnorm(h)
        h_cbam1 = cbam(h_batchnorm1)
        h_cbamAndH1 = h_cbam1 + h
        h_cbamAndH1_ReLU = self.active(h_cbamAndH1)
        rmp = RMPblock(in_channels=h.shape[1])
        h_rmp = rmp(h_batchnorm1)
        h_odc1_ReLU = self.active(h_rmp)
        h_cbamAndRmp = torch.cat([h_cbamAndH1_ReLU, h_odc1_ReLU], 1)
        convlow1 = nn.Conv2d(h_cbamAndRmp.shape[1],h.shape[1],1,1).to(device)
        h_convlowAndRmp = convlow1(h_cbamAndRmp) + h
        h_out1= h_convlowAndRmp + h
        h = h_out1


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
#if __name__ == "__main__":
 #   model = DiffusionUNet()
  #  model.to(device)
    #print(net)
    #input = torch.randn(1, 6, 16, 16).to(device) ### ？
    #t = torch.randn(1, 6, 16, 16).to(device) ### ？
    #Flops, params = profile(model, inputs=(input,t)) # macs
    #print('Flops: % .4fG'%(Flops / 1000000000))# 计算量
    #print('params参数量: % .4fM'% (params / 1000000)) #参数量：等价与上面的summary输出的Total params值
