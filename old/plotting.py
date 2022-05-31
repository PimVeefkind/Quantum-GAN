import numpy as np
import matplotlib.pyplot as plt

def plot_inner_products(n_rounds, inner_products):

    plt.plot( np.arange(n_rounds+1) , inner_products)