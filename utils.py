import torch


def get_schedule(T: int):
    b_t = torch.asarray([10**-4 + 0.02 / T * i for i in range(T)])
    a_t = 1 - b_t
    A_t = torch.asarray([torch.prod(a_t[:t]) for t in range(T)])

    return a_t, b_t, A_t
