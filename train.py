from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from model import MNISTDiffuser


if __name__ == "__main__":
    dataset = MNIST("./MNIST", download=True)

    to_tensor = ToTensor()
    image = to_tensor(dataset[0][0])
    dim = image.shape[1]

    model = MNISTDiffuser(dim)

    print(image)

    output = model(image)
