import tensorflow as tf
from tensorflow import keras

img_size = 32
model = keras.Sequential()
model.add(keras.layers.Conv2D(2*img_size, (3, 3), input_shape=(img_size, img_size, 3), padding='same'))
model.add(keras.layers.LeakyReLU(alpha=0.2))
model.add(keras.layers.Conv2D(4*img_size, (3, 3), strides=(2, 2), padding='same'))
model.add(keras.layers.LeakyReLU(alpha=0.2))
model.add(keras.layers.Conv2D(4*img_size, (3, 3), strides=(2, 2), padding='same'))
model.add(keras.layers.LeakyReLU(alpha=0.2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(1, activation='sigmoid'))

print(model.summary())
model.save('models/discriminator_model.h5')
