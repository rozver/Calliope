from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.backend import mean

# Define Keras integrated loss functions - Binary cross-entropy and Mean Squared Error
bin_cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)
mse = keras.losses.MeanSquaredError()


# Generator DCGAN loss function - Binary cross-entropy
def generator_dcgan_loss_function(fake_prediction):
    generator_loss = bin_cross_entropy(tf.ones_like(fake_prediction), fake_prediction)
    return generator_loss


# Generator LSGAN loss function - Mean Squared Error -> Least Squares Error
def generator_lsgan_loss_function(fake_prediction):
    generator_loss = mse(tf.ones_like(fake_prediction), fake_prediction)
    return generator_loss


# Generator WGAN loss function - Earth mover's distance - Wasserstein loss
def generator_wgan_loss_function(fake_prediction):
    generator_loss = -mean(fake_prediction)
    return generator_loss


# Generate vector noise with random values in shape (batch_size, random_noise_size)
def generate_noise(batch_size, random_noise_size):
    return tf.random.normal([batch_size, random_noise_size])


"""
    Generator architecture - 1 Dense, 4 Conv2DTranspose, 4 BatchNormalization, 4 LeakyReLU
    The first Dense layer is also the input layer and it reshapes the input noise into shape 4x4x256
    Then reshape it until it reaches size 32x32x3 and that is the output of the last layer
    Improving the speed and stability by using BatchNormalization layers
    The Generator architecture is the same for all 3 GAN architectures
"""


class Generator(keras.Model):

    def __init__(self, img_size=32, random_noise_size=100):
        super().__init__(name='generator')

        starting_image_size = int(img_size / 8)

        self.input_layer = keras.layers.Dense(starting_image_size * starting_image_size * 256,
                                              input_shape=(random_noise_size,))
        # self.batch_norm_1 = keras.layers.BatchNormalization()
        self.leaky_1 = keras.layers.LeakyReLU(alpha=0.2)
        self.reshape_1 = keras.layers.Reshape((starting_image_size, starting_image_size, 256))

        self.conv2d_transpose_1 = keras.layers.Conv2DTranspose(starting_image_size * 32, (4, 4), strides=(2, 2),
                                                               padding='same')
        # self.batch_norm_2 = keras.layers.BatchNormalization()
        self.leaky_2 = keras.layers.LeakyReLU(alpha=0.2)

        self.conv2d_transpose_2 = keras.layers.Conv2DTranspose(img_size * 32, (4, 4), strides=(2, 2),
                                                               padding='same')
        # self.batch_norm_3 = keras.layers.BatchNormalization()
        self.leaky_3 = keras.layers.LeakyReLU(alpha=0.2)

        self.conv2d_transpose_3 = keras.layers.Conv2DTranspose(img_size * 32, (4, 4), strides=(2, 2),
                                                               padding='same')
        # self.batch_norm_4 = keras.layers.BatchNormalization()
        self.leaky_4 = keras.layers.LeakyReLU(alpha=0.2)

        self.output_layer = keras.layers.Conv2DTranspose(3, (3, 3),
                                                         padding='same', activation='tanh')

    # The output of the previous layer is the input to the next one
    def call(self, input_tensor):
        x = self.input_layer(input_tensor)
        x = self.leaky_1(x)
        x = self.reshape_1(x)

        x = self.conv2d_transpose_1(x)
        x = self.leaky_2(x)

        x = self.conv2d_transpose_2(x)
        x = self.leaky_3(x)

        x = self.conv2d_transpose_3(x)
        x = self.leaky_4(x)

        x = self.output_layer(x)
        return x


class ConditionalGenerator(keras.Model):
    def __init__(self, img_size=32, num_classes=16, random_noise_size=100):
        super(ConditionalGenerator, self).__init__(name='conditional_generator')

        starting_image_size = int(img_size / 8)

        self.embedding = keras.layers.Embedding(num_classes, 50)
        self.scaling = keras.layers.Dense(starting_image_size*starting_image_size*3)

        self.reshape_1 = keras.layers.Reshape((starting_image_size, starting_image_size, 3))

        self.dense = keras.layers.Dense(starting_image_size * starting_image_size * 256,
                                        input_shape=(random_noise_size,))
        self.batch_norm_1 = keras.layers.BatchNormalization()
        self.leaky_1 = keras.layers.LeakyReLU(alpha=0.2)
        self.reshape_2 = keras.layers.Reshape((starting_image_size, starting_image_size, 256))

        self.merge = keras.layers.Concatenate()

        self.conv2d_transpose_1 = keras.layers.Conv2DTranspose(starting_image_size * 32, (4, 4), strides=(2, 2),
                                                               padding='same')
        self.batch_norm_2 = keras.layers.BatchNormalization()
        self.leaky_2 = keras.layers.LeakyReLU(alpha=0.2)

        self.conv2d_transpose_2 = keras.layers.Conv2DTranspose(img_size * 32, (4, 4), strides=(2, 2),
                                                               padding='same')
        self.batch_norm_3 = keras.layers.BatchNormalization()
        self.leaky_3 = keras.layers.LeakyReLU(alpha=0.2)

        self.conv2d_transpose_3 = keras.layers.Conv2DTranspose(img_size * 32, (4, 4), strides=(2, 2),
                                                               padding='same')
        self.batch_norm_4 = keras.layers.BatchNormalization()
        self.leaky_4 = keras.layers.LeakyReLU(alpha=0.2)

        self.output_layer = keras.layers.Conv2DTranspose(3, (3, 3),
                                                         padding='same', activation='tanh')

    def call(self, random_noise, input_label):
        label = self.embedding(input_label)
        label = self.scaling(label)
        label = self.reshape_1(label)

        x = self.dense(random_noise)
        x = self.batch_norm_1(x)
        x = self.leaky_1(x)
        x = self.reshape_2(x)

        x = self.merge([x, label])

        x = self.conv2d_transpose_1(x)
        x = self.batch_norm_2(x)
        x = self.leaky_2(x)

        x = self.conv2d_transpose_2(x)
        x = self.batch_norm_3(x)
        x = self.leaky_3(x)

        x = self.conv2d_transpose_3(x)
        x = self.batch_norm_4(x)
        x = self.leaky_4(x)

        x = self.output_layer(x)
        return x
