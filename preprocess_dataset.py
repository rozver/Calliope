import os
import cv2
import numpy as np
import pickle
from random import shuffle

IMAGES_DIR = 'dataset/LLD-logo-files/'
images = os.listdir(IMAGES_DIR)
counter = 0

# Two objects - one for the images and another for their file names
images_dataset = []
labels_dataset = []

# Small batch of the dataset we will be using to try to construct the neural network, not to train with.
BATCH_SIZE = 4095

# Iterate through every image in the dataset
for curr_img in images:
    path = os.path.join(IMAGES_DIR, curr_img)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    images_dataset.append(img)
    labels_dataset.append(curr_img)

    # Stop if the desired batch size is reached
    if counter == BATCH_SIZE:
        break
    counter = counter + 1

# Reshape it to a numpy array of size (-1, 128, 128, 1)
images_dataset = np.array(images_dataset).reshape(-1, 128, 128, 1)

# Zip the two datasets and shuffle them
images_and_labels = list(zip(images_dataset, labels_dataset))
shuffle(images_and_labels)

# Unzip the two datasets
images_dataset, labels_dataset = zip(*images_and_labels)

# Save the images dataset
np.save('images_dataset.npy', images_dataset)

# Save the labels dataset
with open('labels_dataset.pickle', 'wb') as pickle_out:
    pickle.dump(labels_dataset, pickle_out)
