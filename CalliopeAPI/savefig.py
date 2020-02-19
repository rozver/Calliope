from matplotlib import pyplot as plt
import cv2
import numpy as np

img = cv2.imread('/home/rozver/Documents/Calliope/CalliopeAPI/dataset/LLD-icons-files/LLD_favicons_clean_png/080770.png')
img = img / 255.0
plt.imshow(img)
plt.savefig('img.png')
