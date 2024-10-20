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
