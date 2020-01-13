import numpy as np
import matplotlib.pyplot as plt
import pickle

# test.npy is the small batch made of 5000 images from the dataset

images = np.load('images_dataset.npy', allow_pickle=True)

labels_dataset = open('labels_dataset.pickle', 'rb')
labels = pickle.load(labels_dataset)
labels_dataset.close()

print(images.shape)
print(labels)

#now images contains all the logos and labels contains the corresponding logo names
