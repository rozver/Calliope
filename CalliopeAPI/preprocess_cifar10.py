from tensorflow import keras
import numpy as np
from random import shuffle

(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

train_images, test_images = train_images / 255.0, test_images / 255.0

shuffle(train_images)
shuffle(test_images)

np.save('dataset/other_datasets/cifar10_train.npy', train_images)
np.save('dataset/other_datasets/cifar10_test.npy', test_images)
