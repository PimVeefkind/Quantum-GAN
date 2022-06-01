import torch
import torch.nn as nn

from .generator_shots import make_generator

class GeneratorCombiner(nn.Module):

    def __init__(self, qdev ,device ,gen_circuit_params ,n_generators, q_delta=1):

        super().__init__()

        self.n_qubits = gen_circuit_params['qub']
        self.n_ancillas = gen_circuit_params['anc']
        self.gen_depth = gen_circuit_params['depth']
        self.n_generators = n_generators
        
        self.device = device
        self.qdev = qdev

        self.quantum_circuit = make_generator(self.qdev, gen_circuit_params)

        self.q_params = nn.ParameterList(
            [
                nn.Parameter(q_delta * torch.rand(self.gen_depth * self.n_qubits * 2), requires_grad=True)
                for _ in range(n_generators)
            ]
        )


    def forward(self, x):
        # Size of each sub-generator output
        patch_size = 2 ** (self.n_qubits - self.n_ancillas)

        # Create a Tensor to 'catch' a batch of images from the for loop. x.size(0) is the batch size.
        images = torch.Tensor(x.size(0), 0).to(self.device)

        # Iterate over all sub-generators
        for params in self.q_params:

            # Create a Tensor to 'catch' a batch of the patches from a single sub-generator
            patches = torch.Tensor(0, patch_size).to(self.device)
            for elem in x:
                q_out = self.partial_measure(elem, params, self.n_qubits, self.n_ancillas).float().unsqueeze(0)
                patches = torch.cat((patches, q_out))

            # Each batch of patches is concatenated with each other to create a batch of images
            images = torch.cat((images, patches), 1)

        return images


    def partial_measure(self, noise, weights,n_qubits,n_ancillas):
        # Non-linear Transform
        probs = self.quantum_circuit(noise, weights)
        probsgiven0 = probs[: (2 ** (n_qubits - n_ancillas))]
        probsgiven0 /= torch.sum(probs)

            # Post-Processing
        probsgiven = probsgiven0 / torch.max(probsgiven0)
        return probsgiven