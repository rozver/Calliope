import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16


# Extract features from each image and add them to a list
def get_feature_list(model, image_dataset):
    feature_list = []
    for img in image_dataset:
        current_feature = model.predict(np.expand_dims(img, axis=0))
        current_feature = np.array(current_feature)
        feature_list.append(current_feature.flatten())
    return feature_list


# Display the images with their corresponding cluster
def show_images(image_dataset, cluster_labels, batch_size=100):
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
    num_clusters = 64
    clustered_labels_location = 'dataset/icons_cluster_labels.npy'

    # Pre-trained model - VGG16
    print('Loading VGG16...')
    model = VGG16(weights='imagenet', include_top=False)
    print('Finished')

    # Load dataset - only images, without labels
    print('Loading dataset...')
    image_dataset = np.load('dataset/icons_dataset.npy', allow_pickle=True)
    print('Finished')

    # Extract features from each image and add them to a list
    print('Getting feature list...')
    feature_list = get_feature_list(model, image_dataset)
    print('Finished')

    # Obtain the corresponding centroid index for each image in the dataset
    # Note: 42 is the answer to everything
    clustered = KMeans(n_clusters=num_clusters, random_state=42).fit(feature_list)

    # Transform the labels into numpy array
    cluster_labels = np.array(clustered.labels_)

    # Serialize the clustered labels
    print('Serializing labels....')
    np.save(clustered_labels_location, cluster_labels)
    print('Finished')

    # Optional: show batch_size image with their corresponding cluster
    # show_images(image_dataset, cluster_labels, batch_size=100)


if __name__ == '__main__':
    main()
