import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import os

dims = [4,16,64]

fig = plt.figure(figsize=(10, 5))

outer = gridspec.GridSpec(4, 1, wspace=0.005, hspace = 0.05)
imag_size = 28
n_examples = 4

reverse_listing = [[],[],[],[]]

for dim in dims:

    filename = os.getcwd() + '/results/pca_dims/images{}.pt'.format(dim)
    result300 = torch.load(filename)[-1]

    for j, im in enumerate(result300[:n_examples]):
        reverse_listing[j].append(im)

for i, images in enumerate(reverse_listing):
    inner = gridspec.GridSpecFromSubplotSpec(1, len(dims),
                subplot_spec=outer[i])

    for j, im in enumerate(images):

        ax = plt.Subplot(fig, inner[j])
        ax.imshow(im.reshape(imag_size,imag_size), cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        if j==0:
            ax.set_ylabel('Example {}'.format(4-i),)
        if i==3:
            ax.set_xlabel('Depth {}'.format(j+1), rotation = 30)
        fig.add_subplot(ax)

fig.suptitle('QGAN performance at iteration 200', fontsize = 20)

plt.savefig( os.getcwd() + '/results/plots/it300_pca_dim.pdf')
plt.savefig( os.getcwd() + '/results/plots/it300_pca_dim.png')
plt.show()