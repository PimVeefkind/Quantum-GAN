import torch
import numpy as np

from torch.nn import BCELoss

bceloss = BCELoss()


def train_cycle(generator, opt_gen, qcirc_param ,discriminator, opt_disc, real_labels , \
                fake_labels, validation_noise, saved_images , dataloader, data_info ,device,\
                iteration_numb):

    for i, (data, _) in enumerate(dataloader):

        # Data for training the discriminator
        data = data.reshape(-1, data_info['image_size'] * data_info['image_size'])
        real_data = data.to(device)

        # Noise follwing a uniform distribution in range [0,pi/2)
        noise = torch.rand(data_info['batch_size'], qcirc_param['tot_qub'], device=device) * np.pi / 2
        fake_data = generator(noise)

        # Training the discriminator
        discriminator.zero_grad()
        outD_real = discriminator(real_data).view(-1)
        outD_fake = discriminator(fake_data.detach()).view(-1)

        errD_real = bceloss(outD_real, real_labels)
        errD_fake = bceloss(outD_fake, fake_labels)
        # Propagate gradients
        errD_real.backward()
        errD_fake.backward()

        errD = errD_real + errD_fake
        opt_disc.step()

        # Training the generator
        generator.zero_grad()
        outD_fake = discriminator(fake_data).view(-1)
        errG = bceloss(outD_fake, real_labels)
        errG.backward()
        opt_gen.step()

        if iteration_numb % 10 == 0:

            print(f'Iteration: {iteration_numb+1}, Discriminator Loss: {errD:0.3f}, Generator Loss: {errG:0.3f}')

            if iteration_numb % 50 == 0:

                test_images = generator(validation_noise).view(8,1,data_info['image_size'] * data_info['image_size']).cpu().detach()
                saved_images.append(test_images)

        return saved_images
            
