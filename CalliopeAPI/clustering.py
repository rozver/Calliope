import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from tensorflow.keras.applications.resnet50 import ResNet50
import os


# Extract features from each image and add them to a list
def get_feature_list(model, images):
    feature_list = []
    for img in images:
        current_feature = model.predict(np.expand_dims(img, axis=0))
        current_feature = np.array(current_feature)
        feature_list.append(current_feature.flatten())
    return feature_list


# Display the images with their corresponding cluster
def inspect_images(image_dataset, cluster_labels, batch_size=100):
    idx = 0
    for i in image_dataset:
        plt.imshow(i)
        plt.xlabel(cluster_labels[idx])
        plt.show()
        if idx == batch_size-1:
            break
        idx = idx+1


def main():
    # Number of images to be shown after clustering, number of different clusters and clustered labels location
    num_clusters = 16
    clustered_labels_location = 'dataset/LLD_icon_numpy/dataset1_labels.npy'
    dataset_location = 'dataset/LLD_icon_numpy/dataset1.npy'

    # Set Tensorflow to run on CPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # Pre-trained model - VGG16
    print('Loading VGG16...')
    model = ResNet50(weights='imagenet', include_top=False)
    print('Finished')

    # Load dataset - only images, without labels
    print('Loading dataset...')
    images = np.load(dataset_location, allow_pickle=True)
    print('Finished')

    # Extract features from each image and add them to a list
    print('Getting feature list...')
    feature_list = get_feature_list(model, images)
    print('Finished')

    # Obtain the corresponding centroid index for each image in the dataset
    # Note: 42 is the answer to everything
    clustered = KMeans(n_clusters=num_clusters, random_state=42).fit(feature_list)

    # Transform the labels into numpy array
    labels = np.array(clustered.labels_)

    # Serialize the clustered labels
    print('Serializing labels....')
    np.save(clustered_labels_location, labels)
    print('Finished')

    # Optional: show batch_size image with their corresponding cluster
    # inspect_images(image_dataset, cluster_labels, batch_size=100)


if __name__ == '__main__':
    main()
