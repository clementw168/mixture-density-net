from typing import Callable

import torch
from torch.utils.data import Dataset


class FunctionDataset(Dataset):
    def __init__(
        self,
        function: Callable[[torch.Tensor], torch.Tensor],
        n_samples: int,
        range: tuple[float, float] = (-0.1, 0.1),
        noise_std: float = 0.1,
        inverse_function: bool = False,
    ):
        """Initialize the dataset.

        Args:
            function (Callable[[float], float]): Function to approximate.
            n_samples (int): Number of points to sample from the function.
            range (tuple[float, float], optional): Range to sample the points from. Defaults to (-0.1, 0.1).
            noise_std (float, optional): Standard deviation of the noise to add to the function. Defaults to 0.1.
            inverse_function (bool, optional): If True, uses the argument of the function as the label. Defaults to False.
        """
        super().__init__()
        self.function = function
        self.n_samples = n_samples
        self.range = range
        self.noise_std = noise_std

        self.x = torch.rand(size=(n_samples,))
        self.x = self.x * (range[1] - range[0]) + range[0]

        self.y = function(self.x)

        if inverse_function:
            self.y, self.x = self.x, self.y

        self.y += torch.randn_like(self.y) * noise_std

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.x[index].reshape((1,)), self.y[index].reshape((1,))


def sinusoide(x: torch.Tensor) -> torch.Tensor:
    return x + 0.3 * torch.sin(2 * torch.pi * x)
