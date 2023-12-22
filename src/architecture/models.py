import torch
from torch import nn

from src.architecture.modules import MixtureDensityHead


class LinearMDN(torch.nn.Module):
    """Fully connected mixture density network."""

    def __init__(
        self,
        input_dimension: int,
        hidden_dims: list[int],
        output_dimension: int,
        n_mixtures: int,
    ):
        """Initialize the network.

        Args:
            input_dimension (int): Dimension of the input.
            output_dimension (int): Dimension of the output.
            n_mixtures (int): Number of mixtures to use in the mixture density head.
        """
        super().__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.n_mixtures = n_mixtures

        hidden_dims = [input_dimension] + hidden_dims
        self.hidden_layers = torch.nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(
                torch.nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            )

        self.hidden_layers.append(
            torch.nn.Linear(hidden_dims[-1], (output_dimension + 2) * n_mixtures)
        )

        self.batch_norm = nn.BatchNorm1d((output_dimension + 2) * n_mixtures)

        print(self.hidden_layers)

        self.mdn_head = MixtureDensityHead(output_dimension, n_mixtures)

    def forward(self, x):
        """Forward pass of the mixture density network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dimension).

        Returns:
            mu (torch.Tensor): Mean of the Gaussian mixture model. Shape (batch_size, n_mixtures, output_dimension).
            sigma (torch.Tensor): Standard deviation of the Gaussian mixture model. Shape (batch_size, n_mixtures).
            pi (torch.Tensor): Mixing coefficients of the Gaussian mixture model. Shape (batch_size, n_mixtures).
        """
        for layer_index in range(len(self.hidden_layers)):
            x = self.hidden_layers[layer_index](x)
            if layer_index != len(self.hidden_layers) - 1:
                x = torch.nn.functional.relu(x)

        # x = self.batch_norm(x)

        return self.mdn_head(x)


class LinearNet(torch.nn.Module):
    """Fully connected network."""

    def __init__(
        self,
        input_dimension: int,
        hidden_dims: list[int],
        output_dimension: int,
    ):
        """Initialize the network.

        Args:
            input_dimension (int): Dimension of the input.
            output_dimension (int): Dimension of the output.
            n_mixtures (int): Number of mixtures to use in the mixture density head.
        """
        super().__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension

        hidden_dims = [input_dimension] + hidden_dims
        self.hidden_layers = torch.nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(
                torch.nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            )

        self.hidden_layers.append(torch.nn.Linear(hidden_dims[-1], output_dimension))

    def forward(self, x):
        """Forward pass of the fully connected network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dimension).

        Returns:
            output (torch.Tensor): Output of the network. Shape (batch_size, output_dimension).

        """
        output = x
        for layer_index in range(len(self.hidden_layers)):
            output = self.hidden_layers[layer_index](output)
            if layer_index != len(self.hidden_layers) - 1:
                output = torch.nn.functional.relu(output)

        return output


class FlattenLinear(LinearNet):
    def __init__(
<<<<<<< HEAD
        self, input_dimension: tuple[int], hidden_dims: list[int], output_dimension: int
=======
        self,
        input_dimension: tuple[int, ...],
        hidden_dims: list[int],
        output_dimension: int,
>>>>>>> main
    ):
        self.input_dimension = input_dimension
        flattened_dimension = 1
        for dim in input_dimension:
            flattened_dimension *= dim

        super().__init__(flattened_dimension, hidden_dims, output_dimension)

    def forward(self, x):
        return super().forward(x.flatten(start_dim=1))


class FlattenLinearMDN(LinearMDN):
    def __init__(
        self,
<<<<<<< HEAD
        input_dimension: tuple[int],
=======
        input_dimension: tuple[int, ...],
>>>>>>> main
        hidden_dims: list[int],
        output_dimension: int,
        n_mixtures: int,
    ):
        self.input_dimension = input_dimension
        flattened_dimension = 1
        for dim in input_dimension:
            flattened_dimension *= dim

        super().__init__(flattened_dimension, hidden_dims, output_dimension, n_mixtures)

    def forward(self, x):
        return super().forward(x.flatten(start_dim=1))


class BasicCNN(torch.nn.Module):
<<<<<<< HEAD
    def __init__(self, input_dimension: tuple[int], output_dimension: int):
        self.sequential = torch.nn.Sequential(
=======
    def __init__(self, input_dimension: tuple[int, ...], output_dimension: int):
        super().__init__()
        self.conv = torch.nn.Sequential(
>>>>>>> main
            torch.nn.Conv2d(input_dimension[0], 16, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3),
<<<<<<< HEAD
            torch.nn.AdaptiveAvgPool2d((1, 1)),
=======
        )
        self.linear = torch.nn.Sequential(
>>>>>>> main
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, output_dimension),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
<<<<<<< HEAD
        return self.sequential(x)
=======
        x = self.conv(x)
        x = torch.mean(x, dim=(2, 3))
        x = self.linear(x)

        return x
>>>>>>> main


class ConvolutionalMDN(BasicCNN):
    def __init__(
        self,
<<<<<<< HEAD
        input_dimension: tuple[int],
        output_dimension: int,
        n_mixtures: int,
    ):
        super().__init__(input_dimension, output_dimension)
=======
        input_dimension: tuple[int, ...],
        output_dimension: int,
        n_mixtures: int,
    ):
        super().__init__(input_dimension, (output_dimension + 2) * n_mixtures)
>>>>>>> main
        self.mdn_head = MixtureDensityHead(output_dimension, n_mixtures)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = super().forward(x)

        return self.mdn_head(x)


<<<<<<< HEAD
class CoughMDN(nn.Module):
    def __init__(self, n_mixtures: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=128 * 5 * 4, out_features=128)
        self.linear2 = nn.Linear(in_features=128, out_features=3*n_mixtures)#3 * n_mixtures)

        self.bn = nn.BatchNorm1d(3*n_mixtures)
        self.mdn_head = MixtureDensityHead(1, n_mixtures)


    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.linear1(x)
        # x = nn.functional.relu(x)
        x = self.linear2(x)
        # Add batch norm
        x = self.bn(x)


        # Return sigmoid
        # return torch.sigmoid(x)
        out = self.mdn_head(x)

        return out

class CoughCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=128 * 5 * 4, out_features=128)
        self.linear2 = nn.Linear(in_features=128, out_features=1)
        self.bn = nn.BatchNorm1d(128)


    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.bn(x)
        x = self.linear2(x)

        return torch.sigmoid(x)

=======
>>>>>>> main
if __name__ == "__main__":
    input_tensor = torch.randn(32, 3)
    model = LinearMDN(
        input_dimension=3, hidden_dims=[32, 32], output_dimension=3, n_mixtures=5
    )

    mu, sigma, pi = model(input_tensor)

    print(mu.shape)
    print(sigma.shape)
    print(pi.shape)
