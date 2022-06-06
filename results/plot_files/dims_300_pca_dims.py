import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

dims = [4,16,64]

fig = plt.figure(figsize=(5, 5))

outer = gridspec.GridSpec(4, 1, wspace=0.005, hspace = 0.05)
imag_size = 28
n_examples = 4

reverse_listing = [[],[],[],[]]

for dim in dims:

    filename = os.getcwd() + '/results/raw_results/pca_dims/images{}.pt'.format(dim)
    result300 = torch.load(filename)[7]

    for j, im in enumerate(result300[:2*n_examples:2]):
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
            ax.set_xlabel('# Dimensions {}'.format(dims[j]), rotation = 15)
        fig.add_subplot(ax)

fig.suptitle('QGAN performance at iteration 350', fontsize = 20)

plt.tight_layout()
plt.savefig( os.getcwd() + '/results/plots/pdf/it300_pca_dim.pdf', bbox_inches='tight')
plt.savefig( os.getcwd() + '/results/plots/png/it300_pca_dim.png', bbox_inches='tight')
plt.show()