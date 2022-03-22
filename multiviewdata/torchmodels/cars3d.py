# Flatten 3D image
import torch
from torch import nn


class Flatten3D(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


# Unflatten to 3D image
class Unflatten3D(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], 64, 4, 4)
        return x


# Increase 2 dims to get a 3D tensor
class Unsqueeze3D(nn.Module):
    def forward(self, x):
        x = x.unsqueeze(-1)
        x = x.unsqueeze(-1)
        return x


# CNNEncoder (the input is fixed to be 64x64xchannels)
class CNNEncoder(nn.Module):
    def __init__(self, z_dim=3, c_dim=1, channels=1):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.pipe = nn.Sequential(
            nn.Conv2d(channels, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            Flatten3D(),
            nn.Linear(4 * 4 * 64, 256),
            nn.ReLU(True)
        )
        # map to shared
        self.S = nn.Linear(256, z_dim)
        # map to private
        self.P = nn.Linear(256, c_dim)

    def forward(self, x):
        tmp = self.pipe(x)
        shared = self.S(tmp)
        private = self.P(tmp)
        return shared, private


# CNNDecoder
class CNNDecoder(nn.Module):
    def __init__(self, z_dim=3, c_dim=1, channels=1):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.pipe = nn.Sequential(
            nn.Linear(z_dim + c_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 1024),
            nn.ReLU(True),
            Unflatten3D(),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, channels, 4, 2, 1),
        )

    def forward(self, s, p):
        # Reconstruct using both the shared and private
        recons = self.pipe(torch.cat((s, p), 1))
        return recons
