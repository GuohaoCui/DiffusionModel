import torch
import torch.nn as nn
import torch.nn.functional as F
import random
#Spatial Transformer Network
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class STNWithCBAM(nn.Module):
    def __init__(self, channels, size):
        super(STNWithCBAM, self).__init__()

        self.localization = nn.Sequential(
            nn.Conv2d(channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(256 * int(size / 2) * int(size / 2), 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 6),
        )

        self.cbam = CBAM(channels)


    def forward(self, x):
        n, c, h, w = x.shape

        theta = self.localization(x)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, (n, c, h, w))
        x = F.grid_sample(x, grid)

        features = self.cbam(x)
        return features

class CBAM(nn.Module):
    def __init__(self, channels):
        super(CBAM, self).__init__()

        self.channels = channels

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.channels, self.channels//4, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels//4, self.channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        # self.relu = stochasticReLU(low=0.57735,high=1.73205)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        attention = self.attention(x)
        x = x * attention
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = x * attention

        return x

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
#         return x3


# class stochasticReLU(nn.Module):
#     def __init__(self, low, high):
#         super().__init__()
#         self.low = low
#         self.high = high
#
#     def forward(self, x):
#         random_number = torch.randint(low=1, high=4, size=(), device='cuda')
#         random_mask = (random_number < 3).expand_as(x) # expand the random number to a boolean tensor
#         x = torch.where(random_mask, F.relu(x), x) # use torch.where instead of if-else
#         dim0, dim1, dim2, dim3 = x.shape
#         w = torch.randint(low=-5, high=5, size=(), device='cuda')
#         w_mask = (w != 0).expand_as(x) # expand the w to a boolean tensor
#         x = torch.where(w_mask, torch.where(x <= w/10, torch.zeros_like(x), x), F.relu(x)) # use torch.where instead of if-else
#         x0 = x
#         x1 = x0.reshape(1, -1)
#         K = torch.empty(dim0, dim1, dim2, dim3, device='cuda')
#         K.uniform_(self.low, self.high)
#         K = K.view(-1)
#         K_tensor = torch.tensor(K).to(device)
#         K1 = K_tensor.reshape(1, -1)
#         w_tensor = torch.tensor(w).to(device) # convert w to a tensor
#         w_tensor = w_tensor.expand_as(K1) # expand w to the same shape as K1
#         x2 = torch.where(w_tensor == 0, x1 * K1, x1 * K1 + w/10) # use torch.where instead of if-else
#         x3 = x2.reshape(dim0, dim1, dim2, dim3)
#
#         return x3


# if __name__ == '__main__':
#     # create a test tensor
#     x = torch.randn(4, 64, 32, 32)
#
#     # instantiate the model
#     model = STNWithCBAM(channels=x.shape[1], size=x.shape[2])
#
#     # make a forward pass with the test tensor
#     output = model(x)
#
#     # check if the output has the same shape as the input
#     assert x.shape == output.shape
#
#
#     print("Test  successfully!")