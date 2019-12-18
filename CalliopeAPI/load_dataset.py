import numpy as np
import matplotlib.pyplot as plt
import pickle


def main():
    images = np.load('dataset/images_dataset.npy', allow_pickle=True)

    labels_dataset = open('dataset/labels_dataset.pickle', 'rb')
    labels = pickle.load(labels_dataset)
    labels_dataset.close()

    print(images.shape)
    print(labels)


if __name__ == '__main__':
    main()
