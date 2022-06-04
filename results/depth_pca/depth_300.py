import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import os

depths = np.arange(1,7)

fig = plt.figure(figsize=(10, 5))

outer = gridspec.GridSpec(4, 1, wspace=0.005, hspace = 0.05)
imag_size = 8
n_examples = 4

reverse_listing = [[],[],[],[]]

for depth in depths:

    filename = os.getcwd() + '/results/depth_pca/images{}.pt'.format(depth)
    result300 = torch.load(filename)[4]

    for j, im in enumerate(result300[:n_examples]):
        reverse_listing[j].append(im)

for i, images in enumerate(reverse_listing):
    inner = gridspec.GridSpecFromSubplotSpec(1, len(depths),
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

plt.savefig( os.getcwd() + '/results/plots/it200_pca.pdf')
plt.savefig( os.getcwd() + '/results/plots/it200_pca.png')
plt.show()