import torch
from torch import nn


class Decoder(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()

        self.hidden_size = hidden_size

        # input image
        # input 13 x 13 x 36

        self.linear = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=60*60),
            nn.ReLU(),
            nn.Linear(in_features=60*60, out_features=60*60*12)
        )

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=12, out_channels=6, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(6),
            # image: 62 x 62 x 6

            nn.ConvTranspose2d(in_channels=6, out_channels=3, kernel_size=3),
            # output: 64 x 64 x 3
        )

    def forward(self, x):
        # x: [B, hidden_size]

        x = self.linear(x)
        x = x.reshape(-1, 12, 60, 60)
        x = self.conv(x)
        x = torch.sigmoid(x).view(-1, 3, 64, 64)

        return x
