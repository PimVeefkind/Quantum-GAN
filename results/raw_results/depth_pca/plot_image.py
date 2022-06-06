import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import os

filename = os.getcwd() + '/results/depth_pca/images4.pt'

results = torch.load(filename)

timesteps = [50,100,150,200,250,300,350,400,450,500]
n_examples = 4

fig = plt.figure(figsize=(10, 5))

outer = gridspec.GridSpec(n_examples, 1, wspace=0.005, hspace = 0.1)
imag_size = 8

reverse_listing = [[],[],[],[]]

for i, images in enumerate(results):
    for j, im in enumerate(images[:n_examples]):
        reverse_listing[j].append(im)

for i, images in enumerate(reverse_listing):
    inner = gridspec.GridSpecFromSubplotSpec(1, len(timesteps),
                subplot_spec=outer[i])

    for j, im in enumerate(images):

        ax = plt.Subplot(fig, inner[j])
        ax.imshow(im.reshape(imag_size,imag_size), cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        if j==0:
            ax.set_ylabel('Example {}'.format(4-i),)
        if i==3:
            ax.set_xlabel('Iteration {}'.format(50+50*j), rotation = 30)
        fig.add_subplot(ax)

fig.suptitle('QGAN training progress with parallel GANs ', fontsize = 20)

plt.savefig( os.getcwd() + '/results/plots/imagedepth1_pca.pdf')
plt.savefig( os.getcwd() + '/results/plots/imagedepth1_pca.png')
plt.show()