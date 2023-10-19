import argparse
import torch
from tqdm import tqdm

import random


from model import MNISTDiffuser
from generator import get_dataloader

# Placeholder
class PlaceholderDataLoader:
    def __next__(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if random.random() < 0.1:
            raise StopIteration
        return torch.tensor([[[[random.random()]*28]*28]]*100), torch.tensor([[[[random.random()]*28]*28]]*100), torch.tensor([[random.randint(0, 1000)]]*100)
    def __iter__(self):
        return self

loss_fn: torch.nn.MSELoss = torch.nn.MSELoss()
    

def train(model: MNISTDiffuser, epochs: int, learning_rate: float, momentum: float, batch: int):

    print(batch)
        
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    data_loader = get_dataloader(batch)
    
    losses: list[float] = []
    for e in range(epochs):
        
        avg_loss = 0.0
        
        for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
            
            # print(f'Batch {i}/{len(data_loader)}', end='\r')
        
            inputs, labels, timesteps = data
            
            optimizer.zero_grad()
            
            outputs = model(inputs, timesteps)
            loss: torch.Tensor = loss_fn(outputs, labels)
            loss.backward()
            
            optimizer.step()
            
            avg_loss = (avg_loss + loss.item()) / (i + 1)
            
        
        losses.append(avg_loss)
        
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
    parser.add_argument('-b', '--batch', default=128, type=int)
    
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        model = MNISTDiffuser(1000, 28)
        if args.load:
            model.load_state_dict(torch.load(args.load))
            model.eval()
            
        train(model, args.epochs, args.learning_rate, args.momentum, args.batch)
        
        torch.save(model.state_dict(), args.out)
    
    
    

if __name__ == '__main__':
    main()
