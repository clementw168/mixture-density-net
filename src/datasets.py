from typing import Callable

import os
import json

import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import torchaudio as ta


class FunctionDataset(Dataset):
    def __init__(
        self,
        function: Callable[[torch.Tensor], torch.Tensor],
        n_samples: int,
        range: tuple[float, float] = (-0.1, 0.1),
        noise_std: float = 0.1,
        inverse_function: bool = False,
        sort: bool = False,
    ):
        """Initialize the dataset.

        Args:
            function (Callable[[float], float]): Function to approximate.
            n_samples (int): Number of points to sample from the function.
            range (tuple[float, float], optional): Range to sample the points from. Defaults to (-0.1, 0.1).
            noise_std (float, optional): Standard deviation of the noise to add to the function. Defaults to 0.1.
            inverse_function (bool, optional): If True, uses the argument of the function as the label. Defaults to False.
            sort (bool, optional): If True, sorts the dataset by the x values. Defaults to False.
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

        if sort:
            self.x, indices = torch.sort(self.x)
            self.y = self.y[indices]

        self.y += torch.randn_like(self.y) * noise_std
        self.x += torch.randn_like(self.x) * noise_std

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.x[index].reshape((1,)), self.y[index].reshape((1,))


def sinusoid(x: torch.Tensor) -> torch.Tensor:
    return x + 0.3 * torch.sin(2 * torch.pi * x)


class MNISTDataset(Dataset):
    def __init__(self, train: bool = True):
        super().__init__()
        self.mnist = MNIST(root="data", download=True, train=train)
        self.transform = ToTensor()

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, index):
        image, label = self.mnist[index]

        image = self.transform(image)
        label = torch.tensor(label)

        return image, label


class CoughDataset(Dataset):
    def __init__(
        self,
        audio_path: str,
        label_path: str,
        transformation,
        target_sample_rate,
        num_samples,
        device = 'cpu',
    ):
        name_set = list(
            set([file for file in os.listdir(audio_path) if file.endswith("wav")])
        )

        self.datalist = name_set
        self.audio_path = audio_path
        self.label_path = label_path
        self.device = device
        self.transformation = transformation.to(device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        if idx >= len(self.datalist):
            raise StopIteration

        audio_file_path = os.path.join(self.audio_path, self.datalist[idx])
        label_file_path = os.path.join(
            self.label_path, self.datalist[idx][:-4] + ".json"
        )

        with open(label_file_path, "r") as f:
            content = json.loads(f.read())
            f.close()

        if "age" not in content:
            return None  # Skip this item
        else:
            label = float(content["age"])

        waveform, sample_rate = ta.load(
            audio_file_path
        )  # (num_channels,samples) -> (1,samples) makes the waveform mono

        waveform = self._resample(waveform, sample_rate)
        waveform = self._mix_down(waveform)
        waveform = self._cut(waveform)
        waveform = self._right_pad(waveform)

        waveform = waveform.to(self.device)
        waveform = self.transformation(waveform)

        return waveform, torch.tensor(label).reshape((1,))

    def _resample(self, waveform, sample_rate):
        resampler = ta.transforms.Resample(sample_rate, self.target_sample_rate)
        return resampler(waveform)

    def _mix_down(self, waveform):
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform

    def _cut(self, waveform):
        if waveform.shape[1] > self.num_samples:
            waveform = waveform[:, : self.num_samples]
        return waveform

    def _right_pad(self, waveform):
        signal_length = waveform.shape[1]
        if signal_length < self.num_samples:
            num_padding = self.num_samples - signal_length
            last_dim_padding = (0, num_padding,)
            waveform = torch.nn.functional.pad(waveform, last_dim_padding)
        return waveform
