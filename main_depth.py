import torch
import pennylane as qml
import numpy as np
import os

from tqdm import tqdm

from scripts.vanilla.generator_combiner import GeneratorCombiner
from scripts.vanilla.discriminator import Discriminator
from scripts.vanilla.train_cycle import train_cycle
from scripts.vanilla.load_data import load_data

'''Head for the experiment that sweeps the depth parameter over the
PCA implementation'''

#General settings
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
N_GD_cycles = 500

#Generator parameters
gen_generators = 4
gen_n_qubits = 5
gen_n_anc_qubits = 1
gen_circuit_depth = 5
gen_circ_param = {'qub': gen_n_qubits, 'anc': gen_n_anc_qubits,\
                  'depth': gen_circuit_depth, }

qdev = qml.device("lightning.qubit", wires=gen_n_qubits)

#importing the data
batch_size = 1
image_size = 8
n_samples = 500
data_info = {'batch_size': batch_size, 'image_size': image_size, 'n_samples': n_samples}

dataloader, dataset = load_data("/datasets/mnist_only0_8x8.csv" ,data_info, gen_circ_param, gen_generators)
dataset_info = {'means': dataset.per_pixel_mean, 'stds': dataset.per_pixel_std}
 
#Initialize generator and discriminator
discriminator = Discriminator(int(np.log2(2**image_size/gen_generators))).to(device)
generator = GeneratorCombiner(qdev, device, gen_circ_param, gen_generators).to(device)

optimizer_gen = torch.optim.SGD(generator.parameters(), lr = 0.3)
optimizer_disc = torch.optim.SGD(discriminator.parameters(), lr = 0.01)

#labels associated with real and fake data
real_labels = torch.full((batch_size,), 1.0, dtype=torch.float, device=device)
fake_labels = torch.full((batch_size,), 0.0, dtype=torch.float, device=device)

# Settings for tracking the progress
validation_noise = torch.rand(64, gen_n_qubits, device=device) * np.pi / 2
train_feedback = {'print': 600, 'save_imag': 50, 'display_imag': 600, 'pix_calc': 1}

#Storage for results
saved_images = []
means = []
stds = []

#Variational parameter
depths = np.arange(1,7)

print('Started training the QGAN...')

for depth in (depths):

    gen_circ_param['depth'] = depth
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
                                    )
                                

    np.savetxt(os.getcwd() + '/results/raw_results/depth/mean{}.txt'.format(depth), means)
    np.savetxt(os.getcwd() + '/results/raw_results/depth/std{}.txt'.format(depth), stds)
    torch.save(saved_images, os.getcwd() + '/results/raw_results/depth/images{}.pt'.format(depth))

#plot_validation_images(saved_images, image_size)
#plot_mean_and_std(means, stds)
    






