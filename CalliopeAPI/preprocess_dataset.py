import os
import cv2
import numpy as np
import pickle
from random import shuffle

IMAGES_DIR = 'dataset/LLD-logo-files/'
images = os.listdir(IMAGES_DIR)
counter = 0

images_dataset = []
labels_dataset = []

BATCH_SIZE = 4095

for curr_img in images:
    path = os.path.join(IMAGES_DIR, curr_img)
    img = cv2.imread(path)
    img = cv2.resize(img, (64, 64))
    images_dataset.append(img)
    labels_dataset.append(curr_img)

    if counter == BATCH_SIZE:
        break
    counter = counter + 1

images_dataset = np.array(images_dataset).reshape(-1, 64, 64, 3)

images_and_labels = list(zip(images_dataset, labels_dataset))
shuffle(images_and_labels)

images_dataset, labels_dataset = zip(*images_and_labels)

np.save('images_dataset.npy', images_dataset)

with open('labels_dataset.pickle', 'wb') as pickle_out:
    pickle.dump(labels_dataset, pickle_out)
