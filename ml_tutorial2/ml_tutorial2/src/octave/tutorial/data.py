"""Functions related to data loading and preprocessing.

===============================================================================
    Copyright (c) 2025 OCTAVE. All rights reserved.

    This is proprietary and confidential software of OCTAVE.
    Unauthorized use, reproduction, or distribution is strictly prohibited.
===============================================================================
"""

import torch
from sklearn import datasets, model_selection
from sklearn.preprocessing import StandardScaler
from torch.utils import data as ptd


def load_data(cfg):
    """Loads the California housing dataset."""
    data = datasets.fetch_california_housing()
    x, y = data.data, data.target

    # Train / val split.
    x_train, x_val, y_train, y_val = model_selection.train_test_split(
        x, y, test_size=cfg.val_size, random_state=cfg.seed
    )

    # Standardize features (important for NN training).
    # scaler = StandardScaler()
    # x_train = scaler.fit_transform(x_train)
    # x_val = scaler.transform(x_val)

    # Convert to tensors and ensure float32 dtype for compatibility with model.
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
    x_val = torch.tensor(x_val, dtype=torch.float32)
    y_val = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32)

    train_ds = ptd.TensorDataset(x_train, y_train)
    val_ds = ptd.TensorDataset(x_val, y_val)

    train_loader = ptd.DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = ptd.DataLoader(val_ds, batch_size=cfg.batch_size)

    return train_loader, val_loader
