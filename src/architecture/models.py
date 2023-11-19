import torch

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

        return self.mdn_head(x)


if __name__ == "__main__":
    input_tensor = torch.randn(32, 3)
    model = LinearMDN(
        input_dimension=3, hidden_dims=[32, 32], output_dimension=3, n_mixtures=5
    )

    mu, sigma, pi = model(input_tensor)

    print(mu.shape)
    print(sigma.shape)
    print(pi.shape)
