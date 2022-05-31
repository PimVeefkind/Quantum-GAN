import torch
import pennylane as qml
import numpy as np
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torchvision



import matplotlib.pyplot as plt

from scripts.generator_combiner import GeneratorCombiner
from scripts.discriminator import Discriminator
from scripts.preprocessing import ZerosDataset
from scripts.train_cycle import train_cycle
from scripts.plotting import plot_validation_images

#General settings

device = torch.device("cpu")
N_GD_cycles = 50

#Generator parameters
gen_generators = 4
gen_n_qubits = 5
gen_n_anc_qubits = 1
gen_circuit_depth = 6
gen_circ_param = {'qub': gen_n_qubits, 'anc': gen_n_anc_qubits,\
                  'depth': gen_circuit_depth, 'tot_qub': gen_n_qubits+gen_n_anc_qubits}


qdev = qml.device("lightning.qubit", wires=gen_n_qubits+gen_n_anc_qubits)

# Enable CUDA device if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



#importing the data
batch_size = 1
image_size = 8
data_info = {'batch_size': batch_size, 'image_size': image_size}

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
dataset = ZerosDataset(csv_file= os.getcwd() + "/datasets/mnist_only0_8x8.csv", imag_size=image_size ,transform=transform)

dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, drop_last=True
)

real_labels = torch.full((batch_size,), 1.0, dtype=torch.float, device=device)
fake_labels = torch.full((batch_size,), 0.0, dtype=torch.float, device=device)

discriminator = Discriminator(image_size).to(device)
generator = GeneratorCombiner(qdev, device, gen_circ_param, gen_generators).to(device)

optimizer_gen = torch.optim.Adam(generator.parameters(), lr = 0.3)
optimizer_disc = torch.optim.Adam(discriminator.parameters(), lr = 0.01)

validation_noise = torch.rand(8, gen_n_qubits, device=device) * np.pi / 2
saved_images = []

print('Started training the QGAN...')

for i in range (N_GD_cycles):

    saved_images = train_cycle(
                                generator = generator,
                                opt_gen = optimizer_gen,
                                qcirc_param = gen_circ_param,
                                discriminator = discriminator,
                                opt_disc = optimizer_disc,
                                real_labels = real_labels,
                                fake_labels = fake_labels,
                                validation_noise = validation_noise,
                                saved_images= saved_images,
                                dataloader = dataloader,
                                data_info = data_info,
                                device = device,
                                iteration_numb= i+1,
                                )

plot_validation_images(saved_images, image_size)
    






