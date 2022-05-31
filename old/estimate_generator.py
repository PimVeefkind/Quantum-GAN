from functools import partial
import numpy as np
import torch

from qiskit.quantum_info import partial_trace
from pennylane.math import frobenius_inner_product

from .qnodes import make_qnodes

def estimate_generator(gen_weights, n_state, N_estimates, dev):

    real_disc_circuit, gen_disc_circuit, disc_circuit = make_qnodes(dev)

    inner_products = []


    for n_estimate in range(N_estimates):

        gen_random_number = 2 * np.pi * torch.rand(1)
        gen_noise_angles = torch.FloatTensor([gen_random_number,0])
        state = disc_circuit(gen_noise_angles, gen_weights, n_state).detach().numpy()

        state0 = np.array(partial_trace(state,[0]))
        state1 = np.array(partial_trace(state,[1]))
        inner_products.append(frobenius_inner_product(state0, state1))


    return(np.mean(inner_products))