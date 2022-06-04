import numpy as np
import pandas as pd

from PIL import Image

data = pd.read_csv('mnist.csv').to_numpy().astype(np.uint8)

data = np.delete(data, np.where(data[:,0]!= 0), axis = 0)

print(data.shape)

labels = data[:,0]
x_data = data[:,1:]



old_size = 28
new_size = 8

transformed_data = np.zeros((data.shape[0], new_size**2+1))
transformed_data[:,0] = labels

for i in range(x_data.shape[0]):

    image = x_data[i,:].reshape(old_size,old_size)
    image_resized = np.array(Image.fromarray(image).resize((new_size,new_size)))

    transformed_data[i,1:] = image_resized.flatten()

transformed_data[:,1:] = transformed_data[:,1:] - np.mean(transformed_data[:,1:]) / np.std(transformed_data[:,1:])

np.savetxt('mnist_only0_8x8.csv', transformed_data)


