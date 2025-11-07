"""Experiment configuration.

===============================================================================
    Copyright (c) 2025 OCTAVE. All rights reserved.

    This is proprietary and confidential software of OCTAVE.
    Unauthorized use, reproduction, or distribution is strictly prohibited.
===============================================================================
"""

import torch


class Config:
    """Stores the configuration for the neural network experiment."""

    input_dim = 8  # California housing dataset has 8 features.
    val_size = 0.2
    batch_size = 64
    lr = 0.01
    epochs = 25
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"
