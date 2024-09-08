import torch
from datasets import ClassLabel
import numpy as np
from torch.utils.data import Dataset


class MMTDataset(Dataset):

    def __init__(self, dir, dataset, labels, transforms, channels):

        self.labels = {label: i  for i, label in enumerate(labels)}
        self.transforms = transforms
        self.dir = dir
        self.channels = channels

        self.mean, self.std = {}, {}
        self.mean["mag"], self.std["mag"] = np.loadtxt(f"{dir}/mean_std.csv", delimiter=',', skiprows=1)
        self.mean["phase"], self.std["phase"] = 90, 90

        features = dataset.features
        features["label"] = ClassLabel(names=labels)
        dataset = dataset.cast(features)
        self.dataset = dataset

        print("Loading data")
        self.load_channels()

        if transforms is not None:
            print("Applying transforms")
            self.apply_transforms(transforms)


    def __len__(self):
        return len(self.dataset)
    
    def _load_data_fun(self, column):
        def fun(x):
            data = np.loadtxt(f"{self.dir}/{x[column]}")
            data[data != 0] = (data[data != 0] - self.mean[column]) / self.std[column]
            x[column] = data.reshape(1, -1)
            return x
        return fun
    
    def load_channels(self):
        for ch in self.channels:
            self.dataset = self.dataset.map(self._load_data_fun(ch)).with_format("torch")
    
    
    def apply_transforms(self, transform):
        # for ch in self.channels:
        #     def fun(x):
        #         x[ch] = transform(x[ch])
        #         return x
        self.dataset = self.dataset.map(transform)
              
     
    def __getitem__(self, index):
        sample = self.dataset[index]
        data = [sample[ch] for ch in self.channels]
        data = np.concatenate(data, axis=0)
        label = sample["label"]

        return data, label