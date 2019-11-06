import numpy as np
import matplotlib.pyplot as plt

# test.npy is the small batch made of 5000 images from the dataset
test_images = np.load('dataset.npy', allow_pickle=True)

images = []
labels = []

for img, label in test_images:
    images.append(img)
    labels.append(label)

#now images contains all the logos and labels contains the corresponding logo names
