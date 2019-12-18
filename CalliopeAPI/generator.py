from tensorflow import keras
import numpy as np
import tensorflow as tf

IMG_SIZE = 128
bin_cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)


def generator_loss_function(generator_output):
    return bin_cross_entropy(tf.ones_like(generator_output), generator_output)


def generate_noise(batch_size, random_noise_size):
    return np.random.uniform(-1, 1, size=(batch_size, random_noise_size))


class Generator(keras.Model):

    def __init__(self, random_noise_size = 100):
        super().__init__(name='generator')

        self.input_layer = keras.layers.Dense(units = random_noise_size)
        self.dense_1 = keras.layers.Dense(units = int(IMG_SIZE/32))
        self.leaky_1 = keras.layers.LeakyReLU(alpha = 0.01)
        self.dense_2 = keras.layers.Dense(units = int(IMG_SIZE/16))
        self.leaky_2 = keras.layers.LeakyReLU(alpha = 0.01)
        self.dense_3 = keras.layers.Dense(units = int(IMG_SIZE/8))
        self.leaky_3 = keras.layers.LeakyReLU(alpha = 0.01)
        self.dense_4 = keras.layers.Dense(units = int(IMG_SIZE/4))
        self.leaky_4 = keras.layers.LeakyReLU(alpha = 0.01)
        self.dense_5 = keras.layers.Dense(units = int(IMG_SIZE/2))
        self.leaky_5 = keras.layers.LeakyReLU(alpha = 0.01)
        self.dense_6 = keras.layers.Dense(units = IMG_SIZE)
        self.leaky_6 = keras.layers.LeakyReLU(alpha = 0.01)
        self.output_layer = keras.layers.Dense(units=IMG_SIZE*IMG_SIZE*3, activation = "tanh")
        self.reshape = keras.layers.Reshape((IMG_SIZE, IMG_SIZE, 3))
        
    def call(self, input_tensor):
        x = self.input_layer(input_tensor)
        x = self.dense_1(x)
        x = self.leaky_1(x)
        x = self.dense_2(x)
        x = self.leaky_2(x)
        x = self.dense_3(x)
        x = self.leaky_3(x)
        x = self.leaky_4(x)
        x = self.leaky_4(x)
        x = self.leaky_5(x)
        x = self.leaky_5(x)
        x = self.leaky_6(x)
        x = self.leaky_6(x)
        x = self.output_layer(x)
        return self.reshape(x)
