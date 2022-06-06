import numpy as np
import matplotlib.pyplot as plt


def compare_pixels(generator, validation_noise,dataset_info,pca_settings,data_info):

    real_mean = dataset_info['means']
    real_std = dataset_info['stds']

    tot_real_mean = np.mean(real_mean)

 

    fake_data = generator(validation_noise).view(64,pca_settings['dim'])
    fake_data = fake_data.cpu().detach().numpy()

    fake_data = dataset_info['inv'](fake_data)

    ratio = tot_real_mean / np.mean(fake_data)
    #fake_data *= ratio

    fake_mean = np.mean(fake_data, axis = 0)
    fake_std = np.std(fake_data, axis = 0)

    mean_dif = np.mean(np.abs(real_mean-fake_mean))
    mean_std = np.mean(np.abs(real_std-fake_std))

    return mean_dif, mean_std

def print_losses(iteration_numb,errD_tot,errG):

    print('Iteration: {}, Discriminator Loss: {}, Generator Loss: {}'\
                    .format(iteration_numb+1,errD_tot,errG))


def draw_images(fake_data, real_data,dataset_info , pca_settings,data_info):

    real_data_pca = real_data.view(1,pca_settings['dim']).cpu().detach().numpy()
    fake_data_pca = fake_data.view(1,pca_settings['dim']).cpu().detach().numpy()

    real_data_true = dataset_info['inv'](real_data_pca).reshape(data_info['image_size'], data_info['image_size'])
    fake_data_true = dataset_info['inv'](fake_data_pca).reshape(data_info['image_size'], data_info['image_size'])

    plt.imshow(fake_data_true , cmap = 'gray', vmin =0 , vmax = 1)
    plt.savefig('test.png')
    plt.imshow(real_data_true , cmap = 'gray', vmin =0 , vmax = 1)
    plt.savefig('real.png')
    


def save_images(generator, validation_noise,dataset_info ,pca_settings,data_info):

    fake_data = generator(validation_noise[:8,:]).view(8,pca_settings['dim'])
    test_images = dataset_info['inv'](fake_data.cpu().detach().numpy())
    
    return test_images
