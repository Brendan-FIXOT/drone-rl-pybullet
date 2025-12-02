import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=2, mode=None, optimizer=optim.Adam, lr=1e-3):
        super(NeuralNetwork, self).__init__()
        if mode not in ["actor", "critic"]:
            raise ValueError("Mode must be either 'actor' or 'critic'")
        self.mode = mode
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.optimizer = optimizer(self.parameters(), lr=lr)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        if self.mode == "critic":
            return x.squeeze(-1)  # Return state values
        elif self.mode == "actor":
            return x # Apply softmax to get action probabilities
        else:
            raise ValueError(f"Unknown mode: {self.mode}")