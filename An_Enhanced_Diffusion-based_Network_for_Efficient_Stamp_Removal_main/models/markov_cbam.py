import random

import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class BasicCBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(BasicCBAM, self).__init__()
        self.channels = channels
        self.reduction = reduction

        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels)

        # Spatial Attention
        self.conv1 = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel Attention
        avg = self.avg_pool(x).view(x.size(0), -1)
        channel_att = self.fc2(self.relu(self.fc1(avg))).view(x.size(0), self.channels, 1, 1)

        # Spatial Attention
        spatial_att = self.sigmoid(self.conv1(x))

        # Apply attention
        x = x * channel_att * spatial_att
        return x

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
            dim0, dim1 = x.shape
            x[x <= 0] = 0
            x0 = x
            x1 = x0.reshape(1, -1)
            K = [random.uniform(self.low, self.high) for _ in range(dim0 * dim1)]
            K_tensor = torch.tensor(K).to(device)
            w = random.randint(-5,5)
            K1 = K_tensor.reshape(1, -1)
            if w==0:
               x2 = x1 * K1
            else:
               x2 = x1 * K1 + w/10
            x3 = x2.reshape(dim0, dim1)

        return x3



class MarkovCBAM(nn.Module):
    def __init__(self, channels, reduction=16, num_layers=3):
        super(MarkovCBAM, self).__init__()
        self.channels = channels
        self.reduction = reduction
        self.num_layers = num_layers

        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1).to(device)
        self.fc1 = nn.Linear(channels, channels // reduction).to(device)
        self.relu = stochasticReLU(low=0.57735,high=1.73205)
        self.fc2 = nn.Linear(channels // reduction, channels).to(device)

        # Spatial Attention
        self.conv1 = nn.Conv2d(channels, 1, kernel_size=1, bias=False).to(device)
        self.sigmoid = nn.Sigmoid().to(device)

        # Markov Chain
        self.transition = nn.Parameter(torch.randn(num_layers, 2, 2)).to(device)

    def forward(self, x):
        # Channel Attention
        avg = self.avg_pool(x).view(x.size(0), -1)
        channel_att = self.fc2(self.relu(self.fc1(avg))).view(x.size(0), self.channels, 1, 1)

        # Spatial Attention
        spatial_att = self.sigmoid(self.conv1(x)).to(device)

        # Markov Chain
        states = [(channel_att, spatial_att)]
        for i in range(self.num_layers - 1):
            c_att, s_att = states[-1]
            transition_prob = F.softmax(self.transition[i], dim=0)
            c_att_next = c_att * transition_prob[0, 0] + s_att * transition_prob[0, 1]
            s_att_next = c_att * transition_prob[1, 0] + s_att * transition_prob[1, 1]
            states.append((c_att_next, s_att_next))

        # Apply attention
        c_att_final, s_att_final = states[-1]
        x = x * c_att_final * s_att_final
        return x
