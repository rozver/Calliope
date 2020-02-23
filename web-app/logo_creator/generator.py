from tensorflow import keras
import tensorflow as tf
# from tensorflow.keras.backend import mean

bin_cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)


@tf.function()
def generator_loss_function(fake_prediction):
    generator_loss = bin_cross_entropy(tf.ones_like(fake_prediction), fake_prediction)
    return generator_loss


def generate_noise(batch_size, random_noise_size):
    return tf.random.normal([batch_size, random_noise_size])


class Generator(keras.Model):

    def __init__(self, img_size=64, random_noise_size=100):
        super().__init__(name='generator')

        self.input_layer = keras.layers.Dense(units=int(img_size/4)*int(img_size/4)*256, use_bias=False, input_shape=(random_noise_size,))
        self.batch_norm_1 = keras.layers.BatchNormalization()
        self.leaky_1 = keras.layers.LeakyReLU()
        self.reshape_1 = keras.layers.Reshape((int(img_size/4), int(img_size/4), 256))
        
        self.conv2d_transpose_1 = keras.layers.Conv2DTranspose(img_size*4, (5, 5), strides=(1, 1), padding='same', use_bias=False)
        self.batch_norm_2 = keras.layers.BatchNormalization()
        self.leaky_2 = keras.layers.LeakyReLU()
        
        self.conv2d_transpose_2 = keras.layers.Conv2DTranspose(img_size*2, (5, 5), strides=(2, 2), padding='same', use_bias=False)
        self.batch_norm_3 = keras.layers.BatchNormalization()
        self.leaky_3 = keras.layers.LeakyReLU()
        
        self.output_layer = keras.layers.Conv2DTranspose(3, (5, 5), strides=(2, 2),
                                                         padding='same', use_bias=False, activation='tanh')

    def call(self, input_tensor):
        x = self.input_layer(input_tensor)
        x = self.batch_norm_1(x)
        x = self.leaky_1(x)
        x = self.reshape_1(x)
        
        x = self.conv2d_transpose_1(x)
        x = self.batch_norm_2(x)
        x = self.leaky_2(x)
        
        x = self.conv2d_transpose_2(x)
        x = self.batch_norm_3(x)
        x = self.leaky_3(x)
        
        x = self.output_layer(x)
        return x
