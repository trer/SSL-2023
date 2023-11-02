import torch
from torch import nn
import math
from utils import get_schedule, device, normalize
import matplotlib.pyplot as plt


def sinusoidal_embedding(n, d):
    # Returns the standard positional embedding
    embedding = torch.zeros(n, d)
    wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])
    wk = wk.reshape((1, d))
    t = torch.arange(n).reshape((n, 1))
    embedding[:, ::2] = torch.sin(t * wk[:, ::2])
    embedding[:, 1::2] = torch.cos(t * wk[:, ::2])

    return embedding


class MNISTDiffuser(torch.nn.Module):
    def __init__(self, n_timesteps, dim, time_emb_dim=100):
        super().__init__()
        # Simple CNN UNet architecture
        # self.pos_embedding = SinusoidalPosEmb(dim)
        self.dim = dim
        # Sinusoidal embedding
        self.time_embed = torch.nn.Embedding(n_timesteps, self.dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_timesteps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        # First half
        self.te1 = self._make_te(time_emb_dim, 1)
        self.n_timesteps = n_timesteps
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
        )
        self.down1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
        )

        self.down2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        # bottle neck

        self.bottleneck = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
        )

        # Up
        self.up1 = torch.nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.block3 = torch.nn.Sequential(
            torch.nn.Conv2d(128 + 64, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
        )

        self.up2 = torch.nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

        self.block4 = torch.nn.Sequential(
            torch.nn.Conv2d(64 + 32, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
        )

        self.final = torch.nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)

    def generate_sample(self):
        ims = []
        with torch.no_grad():
            a, b, _ = get_schedule(self.n_timesteps)

            x = torch.normal(0, 1, size=(1, 1, self.dim, self.dim)).to(device=device)
            for t in range(self.n_timesteps - 1, 1, -1):
                ims.append(x)
                a_t = a[t]
                a_t_bar = a[:t].prod()
                b_t = b[t]
                sigma = torch.sqrt(b_t)
                z = torch.normal(0, 1, size=(1, 1, self.dim, self.dim)).to(
                    device=device
                )

                time_tensor = (torch.ones(1, 1) * t).to(device).long()
                noise = self.forward(x, time_tensor)
                if t == 1:
                    z = torch.zeros_like(z)
                x = (1 / torch.sqrt(a_t)) * (
                    x - b_t / (torch.sqrt(1 - a_t_bar)) * noise
                ) + sigma * z

        return ims

    def forward(self, x: torch.tensor, t: torch.tensor):
        n = len(x)
        t = self.time_embed(t.to(device).long())
        out1 = self.block1(x + self.te1(t).reshape(n, -1, 1, 1))
        out2 = self.block2(self.down1(out1))
        out3 = self.bottleneck(self.down2(out2))
        out4 = self.block3(torch.cat([self.up1(out3), out2], dim=1))
        out5 = self.block4(torch.cat([self.up2(out4), out1], dim=1))
        return self.final(out5)

    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out), nn.SiLU(), nn.Linear(dim_out, dim_out)
        )
