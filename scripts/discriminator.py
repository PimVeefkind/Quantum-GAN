import torch.nn as nn

class Discriminator(nn.Module):
    """Fully connected classical discriminator"""

    def __init__(self, input_size):
        super().__init__()

        self.model = nn.Sequential(
            # Inputs to first hidden layer (num_input_features -> 64)
            nn.Linear(input_size * input_size, 32),
            nn.ReLU(),
            # First hidden layer (64 -> 16)
            nn.Linear(32, 16),
            nn.ReLU(),
            # Second hidden layer (16 -> output)
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)