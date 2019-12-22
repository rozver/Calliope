import cv2
from matplotlib import pyplot as plt

img = cv2.imread('img_path.png', 0)
edges = cv2.Canny(img, 128, 128)

plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.subplot(122)
plt.imshow(edges, cmap='gray')
plt.show()
