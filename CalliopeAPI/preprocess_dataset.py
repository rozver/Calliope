import os
import cv2
import numpy as np
import pickle
from random import shuffle


def main():
    # Define parameters
    images_output_location = 'dataset/images_dataset.npy'
    labels_output_location = 'dataset/labels_dataset.pickle'
    img_size = 64
    batch_size = 4096
    images_dir = 'dataset/LLD-logo-files/'
    images = os.listdir(images_dir)
    counter = 0

    # Define two lists - one for the images and one for the labels
    images_dataset = []
    labels_dataset = []

    # For each image in images_dir
    print('Loading images...')
    for curr_img in images:
        # Load it and resize it
        path = os.path.join(images_dir, curr_img)
        img = cv2.imread(path)
        img = cv2.resize(img, (img_size, img_size))
        # Append the images and the labels to the corresponding list
        images_dataset.append(img)
        labels_dataset.append(curr_img)

        # When the desired batch size is reached stop
        if counter == batch_size-1:
            break
        counter += 1
    print('Finished')

    # Transform the images_datset list into numpy array
    print('Reshaping the images...')
    images_dataset = np.array(images_dataset).reshape(-1, img_size, img_size, 3)
    print('Finished')
    # Zip and shuffle the images and the labels
    print('Shuffling the images and labels...')
    images_and_labels = list(zip(images_dataset, labels_dataset))
    shuffle(images_and_labels)
    # Unzip them
    images_dataset, labels_dataset = zip(*images_and_labels)
    print('Finished')

    # Serialize the images
    print('Starting image serialization...')
    np.save(images_output_location, images_dataset)
    print('Finished')

    # Serialize the labels
    print('Starting labels serialization...')
    with open(labels_output_location, 'wb') as pickle_out:
        pickle.dump(labels_dataset, pickle_out)
    print('Finished')


if __name__ == '__main__':
    main()
