import torch

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_validation_images(results, imag_size):

    fig = plt.figure(figsize=(10, 5))
    outer = gridspec.GridSpec(5, 2, wspace=0.1)

    for i, images in enumerate(results):
        inner = gridspec.GridSpecFromSubplotSpec(1, images.size(0),
                    subplot_spec=outer[i])

        images = torch.squeeze(images, dim=1)
        for j, im in enumerate(images):

            ax = plt.Subplot(fig, inner[j])
            ax.imshow(im.numpy().reshape(imag_size,imag_size), cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            if j==0:
                ax.set_title(f'Iteration {50+i*50}', loc='left')
            fig.add_subplot(ax)

    plt.show()