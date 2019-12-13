from discriminator import Discriminator, discriminator_loss, bin_cross_entropy
import cv2
import numpy as np
from tensorflow import keras

IMG_SIZE = 128

test_img_path = '/home/rozver/Documents/Calliope/dataset/LLD-logo-files/hostiran.png'

img = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (128, 128))

img = np.array(img).reshape(1, 128, 128, 1)
img = img / 255.0

discriminator = Discriminator()
discriminator.compile(optimizer='adam', loss=bin_cross_entropy, metrics=['accuracy'])

discriminator.build(input_shape=(1, IMG_SIZE, IMG_SIZE, 1))
print(discriminator.summary())

inputs = keras.Input(shape=(128, 128, 1))
outputs = discriminator(inputs)
discriminator = keras.Model(inputs=inputs, outputs=outputs)

discriminator.save('models/discriminator_model', save_format='tf')

decision = discriminator(img)
print(decision)

