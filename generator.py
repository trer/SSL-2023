import torch

class Generator:

    def __init__(self, T=1000):
        self.T = T
        self.b_t = torch.asarray([10 ** -4 + 0.02 / T * i for i in range(T)])
        self.a_t = torch.asarray([1 - bt for bt in self.b_t])
        self.A_t = torch.asarray([torch.prod(self.a_t[:t]) for t in range(T)])

    def add_noise(self, image, timestep):
        bt = self.b_t[timestep]
        noise = torch.normal(torch.sqrt(1 - bt) * image, bt)
        image = image + noise
        return image, noise

    def noise_to_timestep(self, image, timestep):
        at = self.A_t[timestep]
        mean = torch.sqrt(at) * image
        noise = torch.normal(mean, (1 - at))
        image = image + noise
        return image, noise

    def create_example(self, image, time_step=None):
        if time_step is None:
            time_step = torch.randint(0, self.T, (1,))
        prev, noise = self.noise_to_timestep(image, time_step)
        after, noise = self.add_noise(prev, time_step)
        return after, noise, time_step + 1
