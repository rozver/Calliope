import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16
import os

# Number of images to be clustered and number of different clusters
BATCH_SIZE = 100
NUM_CLUSTERS = 16

# Pre-trained model - VGG16
model = VGG16(weights='imagenet', include_top=False)
print(model.summary())

# Load dataset - only images, without labels
image_dataset = np.load('dataset/dataset.npy', allow_pickle=True)

# Extract features from each image and add them to a list
def get_feature_list():
    counter = 0
    feature_list = []
    for img in image_dataset:
        current_feature = model.predict(np.expand_dims(img, axis=0))
        current_feature = np.array(current_feature)
        feature_list.append(current_feature.flatten())
        if counter == BATCH_SIZE-1:
            break
        counter = counter+1
    return feature_list


# Cluster the images based on features
feature_list = get_feature_list()
clustered = KMeans(n_clusters=NUM_CLUSTERS, random_state=0).fit(feature_list)

# Get where each image was clustered
clustered_areas = clustered.labels_
print(clustered_areas)


# Display the images with their corresponding cluster
def show_images():
    idx = 0
    for i in image_dataset:
        plt.imshow(i)
        plt.xlabel(clustered_areas[idx])
        plt.show()
        if idx == BATCH_SIZE-1:
            break
        idx = idx+1


show_images()
