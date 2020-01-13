import numpy as np
import matplotlib.pyplot as plt
import pickle


def main():
    print('Loading images...')
    images = np.load('dataset/images_dataset.npy', allow_pickle=True)
    print('Finished')

    print('Load cluster labels...')
    cluster_labels = np.load('dataset/cluster_labels.npy', allow_pickle=True)
    print('Finished')

    print('Loading file names')
    labels_dataset = open('dataset/labels_dataset.pickle', 'rb')
    filename_labels = pickle.load(labels_dataset)
    labels_dataset.close()
    print('Finished')

    print(images.shape)
    print(filename_labels)
    print(cluster_labels)


if __name__ == '__main__':
    main()
