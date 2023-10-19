import argparse
import torch.nn as nn
import torch
import random

# Placeholder class names
class Model(nn.Module):
    def train_one_epoch(self) -> float:
        return random.random()


def train(epochs: int, model: Model):
    losses: list[float] = []
    for i in range(epochs):
        model.train(True)
        
        losses.append(model.train_one_epoch())
        
        print(f'Loss: {losses[-1]:.2f}  Best: {min(losses):.2f}')

def main():
    parser = argparse.ArgumentParser()
    
    # Arguments
    parser.add_argument('mode', choices=['train', 'predict'])
    parser.add_argument('-e', '--epochs', type=int, default=100)
    parser.add_argument('-l', '--load', help='Path to the model to load for further training')
    parser.add_argument('-o', '--out', help='Path to store the model', default='model')
    
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        model = Model()
        if args.load:
            model.load_state_dict(torch.load(args.load))
            model.eval()
            
        train(args.epochs, model)
        
        torch.save(model.state_dict(), args.out)
    
    
    

if __name__ == '__main__':
    main()