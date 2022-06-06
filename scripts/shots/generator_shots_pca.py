import pennylane as qml

def make_generator(dev, gen_circuit_params):
    '''Outer function is to make it easy to call the qnode.'''

    n_qubits = gen_circuit_params['qub']
    gen_depth = gen_circuit_params['depth']


    @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
    def quantum_circuit(noise, weights):
        ''' Code for a single generator implemented in pennylane. 
            The first loop configures the gates introducing the noise
            the second for loop implements the parametrized single
            qubits rotations and entangling layers. The only difference
            with the vanilla implementation is what is returned'''

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

        #SAMPLING INSTEAD OF PROBS!!!
        return qml.sample(wires=list(range(n_qubits)))
    
    return quantum_circuit

