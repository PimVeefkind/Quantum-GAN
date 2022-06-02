import numpy as np
import matplotlib.pyplot as plt
import torch
import os

fig, (ax1,ax2) = plt.subplots(1,2)

depths = np.arange(1,9)
its = np.arange(1,501,10)

for depth in depths:

    mean_file = os.getcwd() + '/results/depth/mean{}.txt'.format(depth)
    std_file = os.getcwd() + '/results/depth/std{}.txt'.format(depth)

    mean = np.loadtxt(mean_file)
    std = np.loadtxt(std_file)

    ax1.plot(its, mean)
    ax2.plot(its, std)
