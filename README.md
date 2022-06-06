# Quantum-GAN
Code for building a quantum generative adversarial network with quantum generator and classical generator. 

### Packages

The directory makes use of several python modules, most notably Pennylane and Pytorch but also the 'standard' python data analysis libraries like numpy and scipy. The easiest way to install them all is by creating a new conda environment using the 'packages.yml' file to populate it with the correct libraries.

### Data
Because of Github size limitations the 'ankleboot' dataset that is used in the 'pca_dim' and 'pca_shot' experiments is not included in the directory. Instead, one should download the MNIST-fashion dataset (for example here: https://www.kaggle.com/datasets/zalando-research/fashionmnist) and remove the other categories manually. The resulting dataset should then be placed in the datasets folder (as .csv file).  

### Code organisation

The code is divided into the following files and folders:
- **datasets**: Mnist zeros and MNIST-fashion datasets, the latter not automatically included (read data section for more info).
- **scripts**: Files holding the data loading, training process and the definition of the generator and discriminator. The generator is fully parametrized and all its parameters can be accessed via the experiment heads (the main_... files). Because the aim of the project was not at optimizing the classical discriminator, its network is hardcoded.
- **main_... files**: Heads that are used to perform the different experiments discussed in the paper. They allow for many different settings.
- **results**: Here the results of the experiments that can be performed via the heads are stored. The 'raw_results' folder are auto-populated with the data collected when running an experimental head. The plot_files can then be used to generate the plots stored in the third folder.
