import torch
from mnist import MNIST
from torch.utils.data import Dataset, DataLoader

from utils import get_schedule, normalize


class MNIST_Dataset(Dataset):

    def __init__(self, T=1000, images=None):
        self.T = T
        self.a_t, self.b_t, self.A_t = get_schedule(T)

        if images is None:
            mndata = MNIST('./MNIST')
            mndata.gz = True
            images, _ = mndata.load_training()
        self.images = images
        self.index = 0
        self.len = len(self.images)

    def __len__(self):
        return self.len

    def __getitem__(self, item) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.index + 1 >= self.len:
            self.index = 0
            raise StopIteration
        img = torch.asarray(self.images[self.index])
        img = normalize(img)
        img = torch.reshape(img, (1, 28, 28))
        self.index = self.index + 1
        return self.create_example(img)

    def add_noise(self, image, timestep):
        abart = self.A_t[timestep]
        noise = torch.randn_like(image)
        image = abart.sqrt() * image + (1 - abart).sqrt() * noise

        return image, noise

    def noise_to_timestep(self, image, timestep):
        at = self.A_t[timestep]
        mean = torch.sqrt(at) * image
        noise = torch.normal(mean, (1 - at))
        image = image + noise
        return image, noise

    def create_example(self, image):
        time_step = torch.randint(0, self.T, (1,))
        after, noise = self.add_noise(image, time_step)
        return after, noise, time_step


def get_dataloader(batch_size):
    dataset = MNIST_Dataset()
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
