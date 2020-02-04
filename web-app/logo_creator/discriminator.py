import tensorflow as tf
from tensorflow import keras
# from tensorflow import reduce_mean as mean

bin_cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss_function(real_prediction, fake_prediction, smoothing_factor=0.9):
    real_loss = bin_cross_entropy(tf.ones_like(real_prediction)*smoothing_factor, real_prediction)
    fake_loss = bin_cross_entropy(tf.zeros_like(fake_prediction), fake_prediction)

    total_loss = real_loss + fake_loss
    return total_loss


"""
@tf.function()
def discriminator_wasserstein_loss_function(real_prediction, fake_prediction):
    real_loss = mean(real_prediction)
    fake_loss = -mean(fake_prediction)

    total_loss = real_loss + fake_loss
    return total_loss
"""

# constraint = keras.constraints.MinMaxNorm(min_value=-0.01, max_value=0.01, rate=1.0, axis=0)


class Discriminator(keras.Model):

    def __init__(self, img_size=64):
        super(Discriminator, self).__init__(name='discriminator')

        self.conv2d_1 = keras.layers.Conv2D(4*img_size, (5, 5),
                                            input_shape=(img_size, img_size, 3), padding='same')
        self.leaky_1 = keras.layers.LeakyReLU()
        self.dropout_1 = keras.layers.Dropout(0.2)
        self.conv2d_2 = keras.layers.Conv2D(8*img_size, (5, 5), padding='same')
        self.leaky_2 = keras.layers.LeakyReLU()
        self.dropout_2 = keras.layers.Dropout(0.2)
        self.flatten = keras.layers.Flatten()
        self.dense_1 = keras.layers.Dense(128, activation='relu')
        self.leaky_3 = keras.layers.LeakyReLU()
        self.dense_2 = keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.conv2d_1(inputs)
        x = self.leaky_1(x)
        x = self.dropout_1(x)
        x = self.conv2d_2(x)
        x = self.leaky_2(x)
        x = self.dropout_2(x)
        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.leaky_3(x)
        x = self.dense_2(x)
        return x
