"""Neural network model definitions.

===============================================================================
    Copyright (c) 2025 OCTAVE. All rights reserved.

    This is proprietary and confidential software of OCTAVE.
    Unauthorized use, reproduction, or distribution is strictly prohibited.
===============================================================================
"""

from torch import nn


class RegressionNet(nn.Module):
    """A simple feed-forward neural network for regression."""

    def __init__(self, input_dim, output_dim=1):
        super().__init__()
        hidden_dim = 64
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        """Implements the forward pass."""
        return self.net(x)
