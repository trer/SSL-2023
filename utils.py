import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_schedule(T: int):
    b_t = torch.linspace(10**-4, 0.02, T)
    a_t = 1 - b_t
    A_t = torch.asarray([torch.prod(a_t[:t]) for t in range(T)])

    return a_t, b_t, A_t


def normalize(img: torch.Tensor, a=-1, b=1) -> torch.Tensor:
    return ((b - a) * (img - img.min()) / (img.max() - img.min())) + a
