from typing import Callable

import os
import json

import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
<<<<<<< HEAD
import torchaudio as ta
=======
>>>>>>> main


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

<<<<<<< HEAD
def kinematic(theta1: torch.Tensor, theta2: torch.Tensor, L1: float = 0.8, L2: float = 0.2) -> torch.Tensor:
    x = L1 * torch.cos(theta1) - L2 * torch.cos(theta1 + theta2)
    y = L1 * torch.sin(theta1) - L2 * torch.sin(theta1 + theta2)
    return torch.cat([x, y], dim=-1)




class MNISTDataset(Dataset):
    def __init__(self, train: bool = True):
        super().__init__()
        self.mnist = MNIST(root="data", download=True, train=train)
=======

class MNISTDataset(Dataset):
    def __init__(self, train: bool = True, classification: bool = False):
        super().__init__()
        self.mnist = MNIST(root="data", download=True, train=train)
        self.classification = classification
>>>>>>> main
        self.transform = ToTensor()

    def __len__(self):
        return len(self.mnist)

<<<<<<< HEAD
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
        # scaler,
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
        # self.scaler = scaler

        # self.labels = []
        # for file in self.datalist:
        #     label_file_path = os.path.join(
        #         self.label_path, file[:-4] + ".json"
        #     )
        #     with open(label_file_path, "r") as f:
        #         content = json.loads(f.read())
        #         f.close()

        #     if "age" not in content:
        #         self.labels.append(None)
        #     else:
        #         self.labels.append(float(content["age"]))

        # self.scaler.fit(self.labels)

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
            # label = float(content["age"])
            label = float(content["cough_detected"])
            # label = self.scaler.transform([[label]])[0][0]
            # Normalize the label to be between 0 and 1 with max 100 min 0
            # label = (label - 0) / (100 - 0)

            # Standardize the label to have mean 0 and std 1
            # label = (label - 50) / 25

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
=======
    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        image, label = self.mnist[index]

        image = self.transform(image)
        label = torch.tensor(label if self.classification else label / 10)
        label = label.reshape(
            1,
        )

        return image, label
>>>>>>> main
