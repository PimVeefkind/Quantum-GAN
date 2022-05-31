import pennylane as qml
import numpy as np
import torch
import os

from .qnodes import make_qnodes

def train_discriminator(N_epochs, N_per_optim_step, n_GD, optimizer, gen_weights, disc_weights,\
                        n_state ,disc_param, dev, save_options, load_options, skip_options):

    real_disc_circuit, gen_disc_circuit, disc_circuit = make_qnodes(dev)


    for n_epoch in range(N_epochs):

        if load_options['allow_loading'] and \
            os.path.exists(load_options['base'] + 'N_GD' + str(n_GD) + 'epoch' + str(n_epoch+1) + '.pt'):

                print('Found precomputed weights on N_GD {} and epoch {} for the discriminator.'.format(n_GD + 1,n_epoch+1))

                disc_weights = torch.load(load_options['base'] + 'N_GD' + str(n_GD) + 'epoch' + str(n_epoch+1) + '.pt')

        else:

            losses = []
            performance_real = []
            performance_fake = []

            for n_step in range(N_per_optim_step):

                real_sample_angles = 2 * np.pi * torch.rand(size = (n_state,2))
                #real_sample_angles = torch.reshape(torch.FloatTensor([0.3,0.4]),(n_state,2))
                real_output = real_disc_circuit(real_sample_angles, disc_weights,n_state, disc_param)
                prob_real_true = (real_output + 1) / 2

                gen_random_number = 2 * np.pi * torch.rand(1)
                gen_noise_angles = torch.FloatTensor([gen_random_number,0])
                #print(qml.draw(gen_disc_circuit)(gen_noise_angles, gen_weights, disc_weights, n_state, disc_param))
                fake_output = gen_disc_circuit(gen_noise_angles, gen_weights, disc_weights,n_state , disc_param)
                prob_fake_true = (fake_output + 1) / 2

                performance_real.append(prob_real_true)
                performance_fake.append(prob_fake_true)
                losses.append(prob_fake_true-prob_real_true)

            loss_total = 1/N_per_optim_step * torch.stack(losses, dim=0).sum(dim=0)

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            if save_options['allow_saving']:

                    torch.save(disc_weights, load_options['base'] + 'N_GD' + str(n_GD) + 'epoch' + str(n_epoch+1) + '.pt')

                    print('succesfully saved discriminator weights!')

            mean_performance_real = 1/N_per_optim_step * torch.mean(torch.stack(performance_real, dim=0).sum(dim=0)).detach().numpy()
            mean_performance_fake = 1/N_per_optim_step * torch.mean(torch.stack(performance_fake, dim=0).sum(dim=0)).detach().numpy()

            print('real performance = {}      generated performance = {}'.format(mean_performance_real, mean_performance_fake))

            if skip_options['allow_skipping'] and mean_performance_real > skip_options['treshold_true']:

                print('Performance of discriminator seems good, exiting round')
                return

            
