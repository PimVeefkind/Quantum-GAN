import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from scipy.signal import savgol_filter

fig, (ax1,ax2) = plt.subplots(2,1, figsize = (6,6), sharex = True)

dims = [4,16,64]
its = np.arange(1,501,1)

for dim in dims:

    mean_file = os.getcwd() + '/results/raw_results/pca_dims/mean{}.txt'.format(dim)
    std_file = os.getcwd() + '/results/raw_results/pca_dims/std{}.txt'.format(dim)

    mean = np.loadtxt(mean_file)
    std = np.loadtxt(std_file)

    ax1.plot(its, savgol_filter(mean,20,3), label = 'd = {}'.format(dim))
    ax2.plot(its, savgol_filter(std,20,3))

ax1.grid(), ax2.grid()
ax1.set_title('$\mathcal{E}\:(F,R)$ as a function of iteration number', fontsize = 14)
ax2.set_title('$\mathcal{V}\:(F,R)$ as a function of iteration number', fontsize = 14)
ax2.set_xlabel('Iteration', fontsize = 14)
ax1.set_ylabel('$\mathcal{E}(F,R)$', fontsize = 16)
ax2.set_ylabel('$\mathcal{V}(F,R)$', fontsize = 16)

ax1.legend( bbox_to_anchor=(0.9, 0.7),loc = 'lower right',ncol = 3 ,fontsize = 12)

plt.savefig(os.getcwd() + '/results/plots/pdf/dim_pca_mstd.png',bbox_inches='tight')
plt.savefig(os.getcwd() + '/results/plots/png/dim_pca_mstd.pdf',bbox_inches='tight')

plt.tight_layout()