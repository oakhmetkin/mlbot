import torch
from torch import nn

from .encoder import Encoder
from .decoder import Decoder


class VAE(nn.Module):
    def __init__(self, hidden_size=80):
        super().__init__()

        self.encoder = Encoder(hidden_size)
        self.decoder = Decoder(hidden_size)

    def gaussian_sampler(self, mu, logsigma):
        if self.training:
            std = torch.exp(0.5 * logsigma)
            eps = torch.randn_like(std)
            sample = mu + (std * eps)
            return sample
        else:
            return mu

    def forward(self, x):
        x = x.float()

        mu, logsigma = self.encoder(x)
        z = self.gaussian_sampler(mu, logsigma)
        reconstruction = self.decoder(z)

        return mu, logsigma, reconstruction


if __name__ == '__main__':
    vae = VAE(hidden_size=50)
    print('VAE module fine')
