import numpy as np
import matplotlib.pyplot as plt
import pickle

images = np.load('images_dataset.npy', allow_pickle=True)

labels_dataset = open('labels_dataset.pickle', 'rb')
labels = pickle.load(labels_dataset)
labels_dataset.close()

print(images.shape)
print(labels)
