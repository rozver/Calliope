import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('img_path.png', 0)
edges = cv2.Canny(img,100,200)

plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.subplot(122)
plt.imshow(edges, cmap='gray')
plt.show()
