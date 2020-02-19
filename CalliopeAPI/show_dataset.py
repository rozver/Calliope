import numpy as np
from matplotlib import pyplot as plt

img = np.load('icon_dataset_2_test.npy', allow_pickle=True)

for i in img:
    plt.imshow(i)
    plt.show()
