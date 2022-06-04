import numpy as np
import torch
import pandas as pd
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torchvision

from torch.utils.data import Dataset

def load_data(csv_name, data_info, circuit_param ,generators):

    assert(data_info['image_size']**2 == generators * 2**(circuit_param['qub']-circuit_param['anc']))

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    dataset = NoPCADataset( os.getcwd() + csv_name ,data_info['n_samples'] , data_info['image_size'],\
                             circuit_param ,generators ,transform=transform)

    dataloader = torch.utils.data.DataLoader(
    dataset, batch_size= data_info['batch_size'], shuffle=True, drop_last=True
    )

    return dataloader, dataset


class NoPCADataset(Dataset):
    """Pytorch dataloader for the Optical Recognition of Handwritten Digits Data Set"""

    def __init__(self, csv_file, n_samples ,imag_size, circuit_param ,generators ,transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv_file = csv_file
        self.imag_size = imag_size
        self.transform = transform

        n_pix_per_gen = 2**(circuit_param['qub']-circuit_param['anc'])

        data = pd.read_csv(self.csv_file, delimiter= ' ').iloc[0:n_samples,:].to_numpy()
        
        for i in range(generators):
            temp = data[:, i*n_pix_per_gen:(i+1)*n_pix_per_gen] + 1e-7
            temp = temp / (np.sum(temp, axis = 1).reshape(data.shape[0],1))
            temp = temp / np.max(temp, axis = 1).reshape(data.shape[0],1)
            data[:, i*n_pix_per_gen:(i+1)*n_pix_per_gen] = temp

        self.per_pixel_mean = np.mean(data, axis = 0)
        self.per_pixel_std = np.mean(data, axis = 0)

        self.df = pd.DataFrame(data)


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.df.iloc[idx, :]
        image = np.array(image)
        image = image.astype(np.float32).reshape(self.imag_size, self.imag_size)

        if self.transform:
            image = self.transform(image)

        # Return image and label
        return image, 0