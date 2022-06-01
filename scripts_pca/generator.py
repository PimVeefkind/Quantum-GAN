import pennylane as qml
import torch

def make_generator(dev, gen_circuit_params):

    n_qubits = gen_circuit_params['qub']
    n_ancillas = gen_circuit_params['anc']
    gen_depth = gen_circuit_params['depth']



    @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
    def quantum_circuit(noise, weights):

        weights = weights.reshape(gen_depth, n_qubits, 2)

        # Initialise latent vectors
        for i in range(n_qubits):
            qml.RY(noise[i], wires=i)

        # Repeated layer
        for i in range(gen_depth):
            # Parameterised layer
            for y in range(n_qubits):
                qml.RY(weights[i][y][0], wires=y)
                qml.RZ(weights[i][y][1], wires=y)

            # Control Z gates
            for y in range(n_qubits - 1):
                qml.CZ(wires=[y, y + 1])

        return qml.probs(wires=list(range(n_qubits)))
    
    return quantum_circuit

