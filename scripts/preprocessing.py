import numpy as np
import torch
import pandas as pd

from torch.utils.data import Dataset


class ZerosDataset(Dataset):
    """Pytorch dataloader for the Optical Recognition of Handwritten Digits Data Set"""

    def __init__(self, csv_file, imag_size = 28 , label=0, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv_file = csv_file
        self.imag_size = imag_size
        self.transform = transform

        self.df = pd.read_csv(self.csv_file, delimiter= ' ')

        self.mean = self.df.mean()
        self.std = self.df.std()
        
        self.df = (self.df - self.mean) / self.std


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.df.iloc[idx, 1:] / 16
        image = np.array(image)
        image = image.astype(np.float32).reshape(self.imag_size, self.imag_size)

        if self.transform:
            image = self.transform(image)

        # Return image and label
        return image, 0