import numpy as np
import pennylane as qml


def real_circuit(angles, n_state):

  for i in range(n_state):
    qml.RY(angles[i,0], wires= i)
    qml.RZ(angles[i,1], wires = i)

    qml.RY(angles[i,0], wires= n_state + i + 1)
    qml.RZ(angles[i,1], wires = n_state + i + 1)

    qml.PauliZ(wires = n_state + i + 1)
    qml.PauliX(wires = n_state + i + 1)

def generator_circuit(noise, theta, n, **kwargs): 
    #shape of theta (n_layers, n_gen, 3)

  wires = np.arange(0,2 * n)

  for i,wire in enumerate(wires):
    qml.RY(noise[i], wires = wire)

  qml.StronglyEntanglingLayers(theta[:-1,:,:], wires = wires)

  for i,wire in enumerate(wires):
    qml.Rot(*theta[-1,i,:], wires = wire)


def discriminator_circuit(theta, n_state, disc_param): 
    #shape of theta (2,3)
  n_disc = disc_param['n']

  disc_wires = range(2 * n_state, 2*n_state+n_disc)
  
  for i in disc_wires:

    qml.Rot(*theta[0,i-2*n_state,:], wires = i)

    for j in range(0,2*n_state):
      qml.CNOT(wires = [j,i])


  qml.StronglyEntanglingLayers(theta[1:,:,:], wires = disc_wires)
