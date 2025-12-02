import torch
import torch.nn as nn


class TransmitMLP(nn.Module):
    """
    MLP that processes genome features to produce modulation factors
    for transmission
    """

    def __init__(self, input_dim=1280, hidden_dim=512, output_dim=1):
        """
        Args:
            input_dim: Dimension of custom_features
            output_dim: 1 outputs: transmission_modifier to multiplies infection probability
        """
        super().__init__()

        self.transmit_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),  # Output in [0, 1]
        )

    def forward(self, features):
        # returns transmission modifier
        return self.transmit_mlp(features)


class ProgressMLP(nn.Module):
    """
    MLP that processes genome features to produce modulation factors
    for disease progression
    """

    def __init__(self, input_dim=1280, hidden_dim=512, output_dim=1):
        """
        Args:
            input_dim: Dimension of custom_features
            output_dim: 1 (multiplies progression probability)
        """
        super().__init__()

        self.progress_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),  # Output in [0, 1]
        )

    def forward(self, features):
        # returns progression modifier
        return self.progress_mlp(features)
