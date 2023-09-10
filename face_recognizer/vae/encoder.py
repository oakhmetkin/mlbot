from torch import nn


class Encoder(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()

        self.hidden_size = hidden_size

        self.conv1 = nn.Sequential(
            # input
            # image: 3 x 64 x 64

            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(6),
            # image: 6 x 62 x 62
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            # image: 12 x 60 x 60
        )

        self.pool1 = nn.MaxPool2d(kernel_size=2)
        # image: 12 x 30 x 30

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(24),
            # image: 24 x 28 x 28
            nn.Conv2d(in_channels=24, out_channels=36, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(36),
            # image: 36 x 26 x 26
        )

        self.pool2 = nn.MaxPool2d(kernel_size=2)
        # image: 36 x 13 x 13

        self.linear = nn.Sequential(
            nn.Linear(in_features=13*13*36, out_features=13*13*12),
            nn.ReLU(),
            # vector: 13*13*12
            nn.Linear(in_features=13*13*12, out_features=2*hidden_size)
            # output: 2*hidden_size
        )

    def forward(self, x):
        # x: [64, 64, 64, 3]
        # x: [BATCH_SIZE, WIDTH, HEIGHT, CHANNELS]
        # x: [B, H, W, C]

        x = x.permute(0, 3, 1, 2)

        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)

        x = x.reshape(-1, 36*13*13)
        x = self.linear(x)

        x = x.view(-1, 2, self.hidden_size)
        mu = x[:, 0, :]
        logsigma = x[:, 1, :]

        return mu, logsigma
