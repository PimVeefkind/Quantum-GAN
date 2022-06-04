import numpy as np
import torch
import pandas as pd
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torchvision

from sklearn.decomposition import PCA

from torch.utils.data import Dataset

def load_data(csv_name, data_info, circuit_param ,pca_dim):

    #assert(np.log2(pca_dim)**2 ==  2**(circuit_param['qub']-circuit_param['anc']))

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    dataset = PCADataset( os.getcwd() + csv_name ,data_info['n_samples'] , data_info['image_size'],\
                             circuit_param ,pca_dim ,transform=transform)

    dataloader = torch.utils.data.DataLoader(
    dataset, batch_size= data_info['batch_size'], shuffle=True, drop_last=True
    )

    return dataloader, dataset


class PCADataset(Dataset):
    """Pytorch dataloader for the Optical Recognition of Handwritten Digits Data Set"""

    def __init__(self, csv_file, n_samples ,imag_size, circuit_param ,pca_dim ,transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv_file = csv_file
        self.imag_size = imag_size
        self.transform = transform
        self.pca_size = int(np.sqrt(pca_dim))
        self.pca_q = int(np.log2(pca_dim))

        n_pix_per_gen = 2**(circuit_param['qub']-circuit_param['anc'])

        data = pd.read_csv(self.csv_file, delimiter= ' ').iloc[0:n_samples,:].to_numpy()

        data = self.make_quantum_suitable(data)
        data_pca = self.perform_pca(data, pca_dim)

        self.min_after_pca = np.min(data_pca)
        data_pca = self.make_quantum_suitable(data_pca - self.min_after_pca)


        assert (np.all(data_pca > 0))

        self.per_pixel_mean = np.mean(data, axis = 0)
        self.per_pixel_std = np.mean(data, axis = 0)

        self.df = pd.DataFrame(data_pca)

    def make_quantum_suitable(self, x):

        x /=  (np.sum(x + 1e-9, axis = 1)).reshape(x.shape[0],1)
        x += 1e-9
        x /= np.max(x, axis = 1).reshape(x.shape[0],1)
        return x

    def reverser(self,x):

        x *= np.max(x, axis = 1).reshape(x.shape[0],1)
        x *=  (np.sum(x + 1e-9, axis = 1)).reshape(x.shape[0],1)
        x += self.min_after_pca
        x = self.pca.inverse_transform(x)
        return x

    def perform_pca(self, x, pca_dim):

        self.pca = PCA(n_components = pca_dim)
        self.pca.fit(x)
        print(self.pca.explained_variance_ratio_)

        return self.pca.transform(x)


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.df.iloc[idx, :]
        image = np.array(image)
        image = image.astype(np.float32).reshape(self.pca_size, self.pca_size)

        if self.transform:
            image = self.transform(image)

        # Return image and label
        return image, 0