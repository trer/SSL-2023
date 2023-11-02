import argparse
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from model import MNISTDiffuser
from generator import MNIST_Dataset, get_dataloader
from utils import device


loss_fn: torch.nn.MSELoss = torch.nn.MSELoss()


def train(
    model: MNISTDiffuser, epochs: int, learning_rate: float, momentum: float, batch: int
):
    model.to(device=device)

    writer = SummaryWriter()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    data_loader = get_dataloader(batch)

    losses: list[float] = []
    for e in range(epochs):
        epoch_losses = []

        for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
            # print(f'Batch {i}/{len(data_loader)}', end='\r')

            inputs, labels, timesteps = [d.to(device=device) for d in data]

            optimizer.zero_grad()

            outputs = model(inputs, timesteps)
            loss: torch.Tensor = loss_fn(outputs, labels)
            loss.backward()

            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        writer.add_scalar("Loss/train (avg)", avg_loss, e)
        losses.append(avg_loss)

        if e % 1 == 0:
            print(f"Epoch {e}  Loss: {losses[-1]:.6f}  Best: {min(losses):.6f}")


def main():
    parser = argparse.ArgumentParser()

    # Arguments
    parser.add_argument("mode", choices=["train", "predict", "debug"])
    parser.add_argument("-e", "--epochs", type=int, default=20)
    parser.add_argument(
        "-l", "--load", help="Path to the model to load for further training"
    )
    parser.add_argument("-o", "--out", help="Path to store the model", default="model")
    parser.add_argument(
        "-r", "--learning-rate", dest="learning_rate", type=float, default=0.001
    )
    parser.add_argument("-m", "--momentum", default=0.9)
    parser.add_argument("-b", "--batch", default=128, type=int)

    args = parser.parse_args()

    if args.mode == "train":
        model = MNISTDiffuser(1000, 28)
        if args.load:
            model.load_state_dict(torch.load(args.load))
            model.eval()

        try:
            train(model, args.epochs, args.learning_rate, args.momentum, args.batch)
        except KeyboardInterrupt:
            pass

        torch.save(model.state_dict(), args.out)
    if args.mode == "predict":
        model = MNISTDiffuser(1000, 28)
        if args.load:
            model.load_state_dict(torch.load(args.load))
            model.eval()

        model.to(device=device)

        samples = model.generate_sample()

        n = 10
        fig, ax = plt.subplots(1, n)
        step_size = len(samples) // n
        for i in range(0, n):
            sample = samples[i * step_size]
            ax[i].imshow(sample.reshape(28, 28).cpu(), cmap="gray")
        plt.imshow(samples[-1].reshape(28, 28).cpu(), cmap="gray")
        plt.savefig("fig")
        plt.figure()
        plt.imshow(sample.reshape(28, 28).cpu(), cmap="gray")
        plt.show()
        # plt.savefig("fig")

    if args.mode == "debug":
        model = MNISTDiffuser(1000, 28)
        if args.load:
            model.load_state_dict(torch.load(args.load, map_location=device))
            model.eval()

        model.to(device=device)
        dataset = MNIST_Dataset()

        image = dataset[0][0].to(device=device).unsqueeze(0)
        print(image)
        noisy, noise = dataset.add_noise(image, 500)

        n = model.forward(noisy, torch.tensor(500))

        plt.figure()
        plt.imshow(noise.detach().reshape(28, 28).cpu(), cmap="gray")
        plt.savefig("fig1")
        plt.figure()
        plt.imshow(n.detach().reshape(28, 28).cpu(), cmap="gray")
        plt.savefig("fig2")

        plt.figure()
        plt.imshow((noise - n).detach().reshape(28, 28).cpu(), cmap="gray")
        plt.show()
        plt.savefig("fig3")


if __name__ == "__main__":
    main()
