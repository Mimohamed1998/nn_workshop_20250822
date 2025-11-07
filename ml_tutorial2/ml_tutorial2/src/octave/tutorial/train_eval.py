"""Neural network training and evaluation for regression tasks.

===============================================================================
    Copyright (c) 2025 OCTAVE. All rights reserved.

    This is proprietary and confidential software of OCTAVE.
    Unauthorized use, reproduction, or distribution is strictly prohibited.
===============================================================================
"""

import torch
from sklearn import metrics
from torch import nn, optim

from octave.tutorial import data, models


def _train_one_epoch(cfg, model, data_loader, loss_fn, optimizer):
    model.train()
    total_loss = 0
    for mini_batch in data_loader:
        x, y = mini_batch
        x, y = x.to(cfg.device), y.to(cfg.device)

        optimizer.zero_grad()
        preds = model(x)
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
    return total_loss / len(data_loader.dataset)


def _evaluate(cfg, model, data_loader):
    model.eval()

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for mini_batch in data_loader:
            X, y = mini_batch
            X, y = X.to(cfg.device), y.to(cfg.device)
            preds = model(X)

            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())

    # Concatenate all predictions and targets.
    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()

    # Compute R^2 using sklearn.
    return metrics.r2_score(targets, preds)


def train_and_eval(cfg):
    """Trains and evaluates a neural network model."""
    torch.manual_seed(cfg.seed)
    train_data_loader, val_data_loader = data.load_data(cfg)

    input_dim = cfg.input_dim
    model = models.RegressionNet(input_dim).to(cfg.device)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    train_losses = []
    train_r2s = []
    val_r2s = []

    for epoch in range(1, cfg.epochs + 1):
        train_loss = _train_one_epoch(cfg, model, train_data_loader, loss_fn, optimizer)
        train_r2 = _evaluate(cfg, model, train_data_loader)
        val_r2 = _evaluate(cfg, model, val_data_loader)

        train_losses.append(train_loss)
        train_r2s.append(train_r2)
        val_r2s.append(val_r2)

        print(
            f"Epoch {epoch:03d}: "
            f"Train Loss = {train_loss:.4f} | Train R^2 = {train_r2:.4f} | "
            f"Val R^2 = {val_r2:.4f}"
        )

        # torch.save(model.state_dict(), "best_model.pt")

    return train_losses, train_r2s, val_r2s
