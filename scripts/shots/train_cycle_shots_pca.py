import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.nn import BCELoss

from .train_feedback_shots_pca import print_losses, draw_images, compare_pixels, save_images

bceloss = BCELoss()


def train_cycle(generator, opt_gen, qcirc_param ,discriminator, opt_disc, real_labels , \
                fake_labels, validation_noise, saved_images, means, stds, train_feedback,\
                dataloader, dataset_info, data_info, device, iteration_numb, pca_settings):

    errD_tot = '-'

    # The for loop auto reads data from the dataloader containing the true ones
    for i, (data, _) in enumerate(dataloader):

        # These are the real samples
        data = data.reshape(-1, pca_settings['size'] * pca_settings['size'])
        real_data = data.to(device)

        # Fake_data is produced from random noise
        noise = torch.rand(data_info['batch_size'], qcirc_param['qub'], device=device) * np.pi / 2
        fake_data = generator(noise)

        # Training the discriminator

        errD_tot = train_discriminator(opt_disc, discriminator, real_data, fake_data, real_labels, fake_labels)
        
        # Training the generator
        errG = train_generator(opt_gen,generator, discriminator, fake_data, real_labels)

        if iteration_numb % train_feedback['print'] == 0:

            print_losses(iteration_numb+1,errD_tot,errG)

        if iteration_numb % train_feedback['save_imag'] == 0:

            saved_images.append(save_images(generator, validation_noise,dataset_info ,pca_settings,data_info))

        if iteration_numb % train_feedback['display_imag'] == 0:

            draw_images(fake_data, real_data, dataset_info,pca_settings,data_info)

        if iteration_numb % train_feedback['pix_calc'] == 0:

            mean, std = compare_pixels( generator, validation_noise, dataset_info, pca_settings,data_info)

            means.append(mean)
            stds.append(std)


        return saved_images, means, stds


def train_discriminator(opt_disc,discriminator,real_data, fake_data,\
                        real_labels, fake_labels):

    discriminator.zero_grad()
    outD_real = discriminator(real_data).view(-1)
    outD_fake = discriminator(fake_data.detach()).view(-1)

    errD_real = bceloss(outD_real, real_labels)
    errD_fake = bceloss(outD_fake, fake_labels)

    errD_real.backward()
    errD_fake.backward()

    errD_tot = errD_real + errD_fake

    opt_disc.step()

    return errD_tot


def train_generator(opt_gen,generator, discriminator, fake_data, real_labels):

    generator.zero_grad()
    outD_fake = discriminator(fake_data).view(-1)
    errG = bceloss(outD_fake, real_labels)
    errG.backward()
    opt_gen.step()

    return errG

            
