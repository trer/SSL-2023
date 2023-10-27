import torch
import math
from utils import get_schedule


def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError(
            "Cannot use sin/cos positional encoding with "
            "odd dimension (got dim={:d})".format(d_model)
        )
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0.0, width).unsqueeze(1)
    pos_h = torch.arange(0.0, height).unsqueeze(1)
    pe[0:d_model:2, :, :] = (
        torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    )
    pe[1:d_model:2, :, :] = (
        torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    )
    pe[d_model::2, :, :] = (
        torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    )
    pe[d_model + 1 :: 2, :, :] = (
        torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    )

    return pe


class MNISTDiffuser(torch.nn.Module):
    def __init__(self, n_timesteps, dim):
        super().__init__()
        # Simple CNN UNet architecture
        # self.pos_embedding = SinusoidalPosEmb(dim)
        self.dim = dim
        self.n_timesteps = n_timesteps
        self.pos_embedding = positionalencoding2d(n_timesteps, dim, dim)
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # decoder that upscales back to the original size
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),
            torch.nn.Sigmoid(),
        )

    def generate_sample(self):
        with torch.no_grad():
            a, b, _ = get_schedule(self.n_timesteps)

            x = torch.normal(0, 1, size=(1, self.dim, self.dim))
            z = torch.normal(0, 1, size=(1, self.dim, self.dim))
            for t in range(self.n_timesteps - 1, 0, -1):
                a_t = a[t]
                a_t_bar = a[:t].prod()
                b_t = b[t]
                sigma = torch.sqrt(b_t)
                noise = self.forward(x, t)
                x = (1 / torch.sqrt(a_t)) * (
                    x - b_t / (torch.sqrt(1 - a_t_bar)) * noise
                ) + sigma * z

        return x

    def forward(self, x, timestep):
        out = x + self.pos_embedding[timestep]
        out = self.encoder(out)
        out = self.decoder(out)
        return out
