from mnist import MNIST
import torch
from torch.utils.data import Dataset, DataLoader
from utils import get_schedule


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
        img = self.images[self.index:self.index + 1]
        img = torch.asarray(img)
        img = torch.reshape(img, (1, 28, 28))
        self.index = self.index + 1
        return self.create_example(img, 1)

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

    def create_example(self, image, batch_size=1):
        time_step = torch.randint(0, self.T, (batch_size,))
        prev, noise = self.noise_to_timestep(image, time_step)
        after, noise = self.add_noise(prev, time_step)
        return after, noise, time_step


def get_dataloader(batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    dataset = MNIST_Dataset()
    train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    for example in train_dataloader:
        print(example[0].shape)
        print(len(example))

