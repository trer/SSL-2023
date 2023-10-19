import argparse
import torch

import random


from model import MNISTDiffuser

# Placeholders
class DataLoader:
    def data(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.tensor([[[[random.random()]*28]*28]]*100), torch.tensor([[[[random.random()]*28]*28]]*100), torch.tensor([[random.randint(0, 1000)]]*100)

loss_fn = torch.nn.MSELoss()
    

def train(model: MNISTDiffuser, epochs: int, learning_rate: float, momentum: float):
        
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    data_loader = DataLoader()
    
    losses: list[float] = []
    for i in range(epochs):
        inputs, labels, timesteps = data_loader.data()
        
        optimizer.zero_grad()
        
        outputs = model(inputs, timesteps)
        loss = loss_fn(outputs, labels)
        loss.backward()
        
        optimizer.step()
        
        losses.append(loss.item())
        
        if i % 10 == 0:
            print(f'Epoch {i}  Loss: {losses[-1]:.2f}  Best: {min(losses):.2f}')
            
        

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