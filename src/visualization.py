import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_mdn_prediction_1d(
    model: torch.nn.Module,
    inference_sample: torch.Tensor,
    label: torch.Tensor,
    device: str,
    n_samples: int = 1000,
):
    model.eval()
    inference_sample = inference_sample.unsqueeze(0).to(device)
    with torch.no_grad():
        mu, sigma, pi = model(inference_sample)

    mu_ = mu.squeeze().cpu().numpy()
    sigma_ = sigma.squeeze().cpu().numpy()
    pi_ = pi.squeeze().cpu().numpy()

    x = np.linspace(-0.1, 1, n_samples)
    x_ = np.broadcast_to(x, (mu_.shape[0], n_samples))

    mu_ = np.broadcast_to(mu_, (n_samples, mu_.shape[0])).T
    sigma_ = np.broadcast_to(sigma_, (n_samples, sigma_.shape[0])).T
    pi_ = np.broadcast_to(pi_, (n_samples, pi_.shape[0])).T

    y = np.exp(-((x_ - mu_) ** 2) / (2 * sigma_**2)) / (sigma_ * np.sqrt(2 * np.pi))
    y = y * pi_
    y = np.sum(y, axis=0)

    plt.imshow(inference_sample.squeeze().cpu().numpy())
    plt.title(f"label {round(label.item()*10)}")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(x * 10, y)
    plt.title("MDN prediction")
    plt.xlabel("Density")
    plt.ylabel("Probability")
    plt.grid()
    plt.xticks(np.arange(0, 11, 1))
    plt.show()
