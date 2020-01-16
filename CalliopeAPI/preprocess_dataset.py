import os
import cv2
import numpy as np
import pickle
from random import shuffle


def main():
    # Define parameters
    images_output_location = 'dataset/icons_dataset.npy'
    img_size = 32
    images_dir = 'dataset/LLD-icons-files/LLD_favicons_clean_png/'
    images = os.listdir(images_dir)

    # Define two lists - one for the images and one for the labels
    images_dataset = []

    # For each image in images_dir
    print('Loading images...')
    for curr_img in images:
        # Load it
        path = os.path.join(images_dir, curr_img)
        img = cv2.imread(path)
        # Append the image to the list of images
        images_dataset.append(img)
    print('Finished')

    # Transform the images_dataset list into numpy array
    print('Reshaping the images...')
    images_dataset = np.array(images_dataset).reshape(-1, img_size, img_size, 3)
    print('Finished')

    # Shuffle the images
    print('Shuffling the images...')
    shuffle(images_dataset)

    # Serialize the images
    print('Starting image serialization...')
    np.save(images_output_location, images_dataset)
    print('Finished')


if __name__ == '__main__':
    main()
