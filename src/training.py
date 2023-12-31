from typing import Any, Callable

import torch
from torch.utils.data import DataLoader


def train_loop(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
    ],
    device: str = "cpu",
) -> float:
    model.train()

    loss_mean = []

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        output = model(x)
        loss = loss_function(output, y)
        loss_mean.append(loss.item())
        loss.backward()
        optimizer.step()

        # print(f"Loss: {loss.item():.4f}", end="\r")

    return sum(loss_mean) / len(loss_mean)


def test_loop(
    model: torch.nn.Module,
    test_loader: DataLoader,
    loss_function: Callable[[Any, torch.Tensor], torch.Tensor],
    device: str,
) -> float:
    model.eval()

    loss_mean = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            output = model(x)
            loss = loss_function(output, y)
            loss_mean.append(loss.item())

    return sum(loss_mean) / len(loss_mean)


def mdn_loss(
    output: tuple[torch.Tensor, torch.Tensor, torch.Tensor], y_true: torch.Tensor
) -> torch.Tensor:
    """Compute the loss of the mixture density network.

    Args:
        output (tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Output of the mixture density network.
            mu (torch.Tensor): Mean of the Gaussian mixture model. Shape (batch_size, n_mixtures, output_dimension).
            sigma (torch.Tensor): Standard deviation of the Gaussian mixture model. Shape (batch_size, n_mixtures).
            pi (torch.Tensor): Mixing coefficients of the Gaussian mixture model. Shape (batch_size, n_mixtures).
        y (torch.Tensor): Target tensor. Shape (batch_size, output_dimension).

    Returns:
        torch.Tensor: Loss of the mixture density network.
    """
    mu, sigma, pi = output

    output_dimension = y_true.shape[1]

    y_true = y_true.unsqueeze(1).repeat(1, mu.shape[1], 1)

    exponent = torch.exp(
        -0.5
        * torch.sum(
            ((y_true - mu) / sigma.unsqueeze(-1).repeat(1, 1, output_dimension)) ** 2,
            dim=-1,
        )
    )

    normalizer = torch.sqrt((2 * torch.pi * sigma**2) ** output_dimension) + 1e-10
    loss = -torch.log((exponent * pi / normalizer + 1e-10).sum(dim=1))
    loss = torch.mean(loss)

    return loss


def mse_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Compute the MSE loss.

    Args:
        y_pred (torch.Tensor): Predicted tensor. Shape (batch_size, output_dimension).
        y_true (torch.Tensor): Target tensor. Shape (batch_size, output_dimension).

    Returns:
        torch.Tensor: MSE loss.
    """

    return torch.mean((y_pred - y_true) ** 2)
