import pennylane as qml
import numpy as np

from .RGD_circuits import real_circuit,generator_circuit, discriminator_circuit

def make_qnodes(dev):


    @qml.qnode(dev, interface="torch")
    def real_disc_circuit(real_angles, theta_disc, n_state, disc_param):

        real_circuit(real_angles, n_state)
        discriminator_circuit(theta_disc, n_state, disc_param)

        return qml.expval(qml.PauliZ(2*n_state))

    @qml.qnode(dev, interface="torch")
    def gen_disc_circuit(noise, theta_gen, theta_disc,n_state , disc_param):

        generator_circuit(noise, theta_gen,n_state)
        discriminator_circuit(theta_disc, n_state, disc_param)

        return qml.expval(qml.PauliZ(n_state))

    @qml.qnode(dev, interface="torch")
    def disc_circuit(noise, theta_gen, n_state):

        generator_circuit(noise, theta_gen,n_state)

        return(qml.density_matrix(wires = np.arange(2*n_state)))

    return real_disc_circuit, gen_disc_circuit, disc_circuit
