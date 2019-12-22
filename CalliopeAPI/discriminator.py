import tensorflow as tf
from tensorflow import keras

bin_cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss_function(real, fake, smoothing_factor=0.9):
    real_loss = bin_cross_entropy(tf.ones_like(real)*smoothing_factor, real)
    fake_loss = bin_cross_entropy(tf.zeros_like(fake), fake)

    total_loss = real_loss + fake_loss
    return total_loss


class Discriminator(keras.Model):

    def __init__(self, img_size=64):
        super(Discriminator, self).__init__(name='discriminator')

        self.conv2d_1 = keras.layers.Conv2D(int(img_size/2), (3, 3), input_shape=(img_size, img_size, 3), activation='relu')
        self.max_pooling2d_1 = keras.layers.MaxPooling2D((2, 2))
        self.dropout_1 = keras.layers.Dropout(0.2)
        self.conv2d_2 = keras.layers.Conv2D(2*img_size, (3, 3), activation='relu')
        self.max_pooling2d_2 = keras.layers.MaxPooling2D((2, 2))
        self.dropout_2 = keras.layers.Dropout(0.2)
        self.conv2d_3 = keras.layers.Conv2D(2*img_size, (3, 3), activation='relu')
        self.max_pooling2d_3 = keras.layers.MaxPooling2D((2, 2))
        self.dropout_3 = keras.layers.Dropout(0.2)
        self.global_average_pooling2d = keras.layers.GlobalAveragePooling2D()
        self.flatten = keras.layers.Flatten()
        self.dense_1 = keras.layers.Dense(128, activation='relu')
        self.dense_2 = keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.conv2d_1(inputs)
        x = self.max_pooling2d_1(x)
        x = self.dropout_1(x)
        x = self.conv2d_2(x)
        x = self.max_pooling2d_2(x)
        x = self.dropout_2(x)
        x = self.conv2d_3(x)
        x = self.max_pooling2d_3(x)
        x = self.dropout_3(x)
        x = self.global_average_pooling2d(x)
        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.dense_2(x)
        return x
