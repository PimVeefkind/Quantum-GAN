import torch
import pennylane as qml
import numpy as np
import os

from tqdm import tqdm

from scripts.generator_combiner import GeneratorCombiner
from scripts.discriminator import Discriminator
from scripts_pca.load_data_pca import load_data
from scripts_pca.train_cycle_pca import train_cycle
from scripts_pca.plotting_pca import plot_validation_images, plot_mean_and_std

#General settings
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
N_GD_cycles = 400

#PCA settings
pca_dim  = 16
pca_q = int(np.log2(pca_dim))
pca_size = int(np.sqrt(pca_dim))
pca_settings = {'dim': pca_dim, 'q': pca_q, 'size': pca_size}

#Generator parameters
gen_generators = 1
gen_n_anc_qubits = 1
gen_n_qubits = pca_q + gen_n_anc_qubits
gen_circuit_depth = 5
gen_circ_param = {'qub': gen_n_qubits, 'anc': gen_n_anc_qubits,\
                  'depth': gen_circuit_depth, }

qdev = qml.device("lightning.qubit", wires=gen_n_qubits)


#importing the data
batch_size = 1
image_size = 28
n_samples = 200
data_info = {'batch_size': batch_size, 'image_size': image_size, 'n_samples': n_samples}

dataloader, dataset = load_data("/datasets/mnist_only0_8x8.csv" ,data_info, gen_circ_param, pca_dim)
dataset_info = {'means': dataset.per_pixel_mean, 'stds': dataset.per_pixel_std, 'inv': dataset.reverser}
 
#Initialize generator and discriminator
discriminator = Discriminator(pca_q).to(device)
generator = GeneratorCombiner(qdev, device, gen_circ_param, gen_generators).to(device)

optimizer_gen = torch.optim.SGD(generator.parameters(), lr = 0.3)
optimizer_disc = torch.optim.SGD(discriminator.parameters(), lr = 0.01)

#labels associated with real and fake data
real_labels = torch.full((batch_size,), 1.0, dtype=torch.float, device=device)
fake_labels = torch.full((batch_size,), 0.0, dtype=torch.float, device=device)

# Settings for tracking the progress
validation_noise = torch.rand(64, gen_n_qubits, device=device) * np.pi / 2
train_feedback = {'print': 600, 'save_imag': 50, 'display_imag': 600, 'pix_calc': 1}

#Variational parameter
depths = np.arange(1,7)

print('Started training the QGAN...')

for depth in (depths):

    gen_circ_param['depth'] = depth

    #Storage for results
    saved_images = []
    means = []
    stds = []

    for i in tqdm(range(N_GD_cycles)):

        saved_images, means, stds = train_cycle(

                                    generator = generator,
                                    opt_gen = optimizer_gen,
                                    qcirc_param = gen_circ_param,
                                    discriminator = discriminator,
                                    opt_disc = optimizer_disc,
                                    real_labels = real_labels,
                                    fake_labels = fake_labels,
                                    validation_noise = validation_noise,
                                    saved_images= saved_images,
                                    means = means,
                                    stds = stds,
                                    train_feedback= train_feedback,
                                    dataloader = dataloader,
                                    dataset_info = dataset_info,
                                    data_info = data_info,
                                    device = device,
                                    iteration_numb= i+1,
                                    pca_settings= pca_settings
                                    )
                                

    np.savetxt(os.getcwd() + '/results/depth_pca/mean{}.txt'.format(depth), means)
    np.savetxt(os.getcwd() + '/results/depth_pca/std{}.txt'.format(depth), stds)
    torch.save(saved_images, os.getcwd() + '/results/depth_pca/images{}.pt'.format(depth))

