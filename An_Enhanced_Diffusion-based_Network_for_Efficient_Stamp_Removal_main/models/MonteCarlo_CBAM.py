# 导入必要的库
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义CBAM模块
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        # 通道注意力模块
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        # 空间注意力模块
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 应用通道注意力
        channel_weight = self.channel_attention(x)
        x = x * channel_weight
        # 应用空间注意力
        spatial_weight = self.spatial_attention(torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]], dim=1))
        x = x * spatial_weight
        return x

# 定义神经网络模型
# class NeuralNetwork(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(NeuralNetwork, self).__init__()
#         # 定义全连接层
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, output_size)
#         # 定义CBAM模块
#         self.cbam = CBAM(hidden_size)
#
#     def forward(self, x):
#         # 前向传播
#         x = F.relu(self.fc1(x))
#         x = self.cbam(x)
#         x = F.softmax(self.fc2(x), dim=1)
#         return x

# 定义蒙特卡洛方法预测注意力权重的函数
    def mc_predict(model, input):
        # 设置模型为评估模式
        model.eval()
        # 初始化注意力权重列表
        attention_weights = []
        # 进行多次随机前向传播
        for _ in range(10):
            # 复制输入张量并添加高斯噪声
            input_noisy = input.clone().detach() + torch.randn_like(input) * 0.01
            # 前向传播并获取CBAM模块的输出
            # cbam_output = model.cbam(model.fc1(input_noisy))
            cbam_output = model.cbam(input_noisy)
            # 计算空间注意力权重并添加到列表中
            spatial_weight = model.cbam.spatial_attention(torch.cat([cbam_output.mean(1, keepdim=True), cbam_output.max(1, keepdim=True)[0]], dim=1))
            attention_weights.append(spatial_weight.squeeze().detach().numpy())
        # 将注意力权重列表转换为numpy数组并求平均值
        attention_weights = np.array(attention_weights).mean(axis=0)
        # attention_weights = attention_weights.reshape(input.shape[0],input.shape[1],input.shape[2],input.shape[3])
        # print(attention_weights.shape)
        return attention_weights




# # 创建一个神经网络模型实例
# model = NeuralNetwork(784, 256, 10)
# # 创建一个随机输入张量
# input = torch.randn(1,256, 1,784)
# # 调用蒙特卡洛方法预测注意力权重的函数并打印结果
# attention_weights = mc_predict(model, input)
# model2 = CBAM(channels=256,reduction=16)
# out = model2(input)
# print("out",out)
# print("attention_weights",attention_weights)
# attention_weights = torch.from_numpy(attention_weights)
# output = 0.1 * attention_weights * out + 0.9 * out
# print("output",output)