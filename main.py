import argparse
import torch

import random


from model import MNISTDiffuser
from generator import Generator

# Placeholder
class PlaceholderDataLoader:
    def __next__(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if random.random() < 0.1:
            raise StopIteration
        return torch.tensor([[[[random.random()]*28]*28]]*100), torch.tensor([[[[random.random()]*28]*28]]*100), torch.tensor([[random.randint(0, 1000)]]*100)
    def __iter__(self):
        return self

loss_fn = torch.nn.MSELoss()
    

def train(model: MNISTDiffuser, epochs: int, learning_rate: float, momentum: float):
        
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    data_loader = Generator()
    
    losses: list[float] = []
    for e in range(epochs):
        
        for i, data in enumerate(data_loader):
            
            print(f'Batch {i}', end='\r')
        
            inputs, labels, timesteps = data
            
            print(inputs.shape)
            print(labels.shape)
            print(timesteps.shape)
            
            optimizer.zero_grad()
            
            outputs = model(inputs, timesteps)
            loss = loss_fn(outputs, labels)
            loss.backward()
            
            optimizer.step()
        
        losses.append(loss.item())
        
        if e % 1 == 0:
            print(f'Epoch {e}  Loss: {losses[-1]:.2f}  Best: {min(losses):.2f}')
            
        

def main():
    parser = argparse.ArgumentParser()
    
    # Arguments
    parser.add_argument('mode', choices=['train', 'predict'])
    parser.add_argument('-e', '--epochs', type=int, default=100)
    parser.add_argument('-l', '--load', help='Path to the model to load for further training')
    parser.add_argument('-o', '--out', help='Path to store the model', default='model')
    parser.add_argument('-lr', '--learning-rate', dest='learning_rate', default=0.001)
    parser.add_argument('-m', '--momentum', default=0.9)
    
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        model = MNISTDiffuser(1000, 28)
        if args.load:
            model.load_state_dict(torch.load(args.load))
            model.eval()
            
        train(model, args.epochs, args.learning_rate, args.momentum)
        
        torch.save(model.state_dict(), args.out)
    
    
    

if __name__ == '__main__':
    main()