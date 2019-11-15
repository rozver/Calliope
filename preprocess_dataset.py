import os
import cv2
import numpy as np

IMAGES_DIR = 'dataset/LLD-logo-files/'
images = os.listdir(IMAGES_DIR)
counter = 0
images_with_file_names = []

# Small batch of the dataset we will be using to try to construct the neural network, not to train with.
BATCH_SIZE = 4999

for curr_img in images:
    path = os.path.join(IMAGES_DIR, curr_img)
    img = cv2.imread(path)
    img = cv2.resize(img, (128, 128))
    images_with_file_names.append(img)
    if counter == BATCH_SIZE:
        break
    counter = counter + 1

images_with_file_names = np.array(images_with_file_names).reshape(-1, 128, 128, 3)

with open('dataset.npy', 'wb') as output_file:
    np.save(output_file, images_with_file_names)
