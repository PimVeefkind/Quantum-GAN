import numpy as np
import matplotlib.pyplot as plt

'''File contains function for various data displaying/storing options that are called
from the train_cycle.py file.'''

def compare_pixels(dataset_info, generator, validation_noise, data_info):
    '''This calculates and stores the mean of mean/std of absolute pixel difference
    between the fake and real data. Although it is not a perfect measure of performance
    it does give some general inside.'''

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
    '''print loss of the generator and discriminator. Really not
    that informative, is useful for monitoring divergence.'''

    print('Iteration: {}, Discriminator Loss: {}, Generator Loss: {}'\
                    .format(iteration_numb+1,errD_tot,errG))


def draw_images(fake_data, real_data, size):
    '''Function that can be used to display a fake and real sample from
    the training process during training. This is really useful for monitoring
    altough it is quite time consuming because it has to make a png image. 
    Making this image instead of showing does make it useable also on ssh-connected
    computers which is why this implementation is chosen.'''

    fake_data = fake_data.view(1,size * size)
    real_data = real_data.view(1,size * size)

    plt.imshow(fake_data.cpu().detach().numpy().reshape(size,size) , cmap = 'gray')
    plt.savefig('test.png')
    plt.imshow(real_data.cpu().detach().numpy().reshape(size,size) , cmap = 'gray')
    plt.savefig('real.png')
    


def save_images(generator, validation_noise, data_info):
    '''Stores generator created images'''

    fake_data = generator(validation_noise[:8,:]).view(8,data_info['image_size'] * data_info['image_size'])
    test_images = fake_data.cpu().detach()
    
    return test_images
