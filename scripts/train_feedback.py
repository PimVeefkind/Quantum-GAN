import numpy as np
import matplotlib.pyplot as plt


def compare_pixels(dataset_info, generator, validation_noise, data_info):

    real_mean = dataset_info['means']
    real_std = dataset_info['stds']

    fake_data = generator(validation_noise).view(64,data_info['image_size'] * data_info['image_size'])
    fake_data = fake_data.cpu().detach().numpy()

    fake_mean = np.mean(fake_data, axis = 0)
    fake_std = np.std(fake_data, axis = 0)

    mean_dif = np.mean(np.abs(real_mean-fake_mean))
    mean_std = np.mean(np.abs(real_std-fake_std))

    return mean_dif, mean_std

def print_losses(iteration_numb,errD_tot,errG):

    print('Iteration: {}, Discriminator Loss: {}, Generator Loss: {}'\
                    .format(iteration_numb+1,errD_tot,errG))


def draw_images(fake_data, real_data, size):

    fake_data = fake_data.view(1,size * size)
    real_data = real_data.view(1,size * size)

    plt.imshow(fake_data.cpu().detach().numpy().reshape(size,size) , cmap = 'gray', vmin =0 , vmax = 1)
    plt.savefig('test.png')
    plt.imshow(real_data.cpu().detach().numpy().reshape(size,size) , cmap = 'gray', vmin =0 , vmax = 1)
    plt.savefig('real.png')
    


def save_images(generator, validation_noise, data_info):

    fake_data = generator(validation_noise[:8,:]).view(8,data_info['image_size'] * data_info['image_size'])
    test_images = fake_data.cpu().detach()
    
    return test_images
