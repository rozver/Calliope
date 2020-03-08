from image_noise import add_noise
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

image = Image.open('D:/Programs/Solutions/My Projects/Calliope/Enhancing/images/1.png').convert('RGBA')
data = np.asarray(image)

plt.figure(figsize=(18,24))
plt.imshow(add_noise(data, "gaussian"))
plt.show()