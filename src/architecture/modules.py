import torch


class MixtureDensityHead(torch.nn.Module):
    """Mixture density head. The input should have a size of (output_dimension + 2) * n_mixtures

    Args:
        n_mixtures (int): Number of mixtures to use in the mixture density head.
    """

    def __init__(self, output_dimension: int, n_mixtures: int):
        super().__init__()
        self.output_dimension = output_dimension
        self.n_mixtures = n_mixtures

    def forward(self, x):
        """Forward pass of the mixture density head.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, (output_dimension + 2) * n_mixtures).

        Returns:
            mu (torch.Tensor): Mean of the Gaussian mixture model. Shape (batch_size, n_mixtures, output_dimension).
            sigma (torch.Tensor): Standard deviation of the Gaussian mixture model. Shape (batch_size, n_mixtures).
            pi (torch.Tensor): Mixing coefficients of the Gaussian mixture model. Shape (batch_size, n_mixtures).
        """

        mu = x[:, : self.output_dimension * self.n_mixtures].reshape(
            -1, self.n_mixtures, self.output_dimension
        )
        sigma = torch.exp(
            x[
                :,
                -2 * self.n_mixtures : -self.n_mixtures,
            ]
        )

        pi = torch.nn.functional.softmax(x[:, -self.n_mixtures :], dim=1)

        return mu, sigma, pi
