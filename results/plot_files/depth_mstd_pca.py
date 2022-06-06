import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from scipy.signal import savgol_filter

fig, (ax1,ax2) = plt.subplots(2,1, figsize = (6,6), sharex = True)

depths = np.arange(1,7)
its = np.arange(1,401,1)

for depth in depths:

    mean_file = os.getcwd() + '/results/raw_results/depth_pca/mean{}.txt'.format(depth)
    std_file = os.getcwd() + '/results/raw_results/depth_pca/std{}.txt'.format(depth)

    mean = np.loadtxt(mean_file)
    std = np.loadtxt(std_file)

    ax1.plot(its, savgol_filter(mean,20,3))
    ax2.plot(its, savgol_filter(std,20,3), label = 'd = {}'.format(depth))

ax1.grid(), ax2.grid()
ax1.set_title('$\mathcal{E}\:(F,R)$ as a function of iteration number', fontsize = 14)
ax2.set_title('$\mathcal{V}\:(F,R)$ as a function of iteration number', fontsize = 14)
ax2.set_xlabel('Iteration', fontsize = 14)
ax1.set_ylabel('$\mathcal{E}(F,R)$', fontsize = 16)
ax2.set_ylabel('$\mathcal{V}(F,R)$', fontsize = 16)

ax2.legend( bbox_to_anchor=(1.25, -0.06),loc = 'lower right', fontsize = 12)

plt.savefig(os.getcwd() + '/results/plots/pdf/depthmstd_pca.png',bbox_inches='tight')
plt.savefig(os.getcwd() + '/results/plots/png/depthmstd_pca.pdf',bbox_inches='tight')

plt.tight_layout()