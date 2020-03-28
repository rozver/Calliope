import tensorflow as tf
from tensorflow import keras
from tensorflow import reduce_mean as mean
from constraint import ClipConstraint

# Define Keras integrated loss functions - Binary cross-entropy and Mean Squared Error
bin_cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)
mse = keras.losses.MeanSquaredError()

# Define Critic constraint - weight clipping
constraint = ClipConstraint(0.01)


# Loss function for DCGAN - Binary cross-entropy
def discriminator_dcgan_loss_function(real_prediction, fake_prediction, smoothing_factor=0.9):
    real_loss = bin_cross_entropy(tf.ones_like(real_prediction)*smoothing_factor, real_prediction)
    fake_loss = bin_cross_entropy(tf.zeros_like(fake_prediction), fake_prediction)

    total_loss = real_loss + fake_loss
    return total_loss


# Loss function for LSGAN - Mean Squared Error -> Least Squares
def discriminator_lsgan_loss_function(real_prediction, fake_prediction, smoothing_factor=0.9):
    real_loss = mse(tf.ones_like(real_prediction), real_prediction)
    fake_loss = mse(tf.zeros_like(fake_prediction), fake_prediction)

    total_loss = real_loss + fake_loss
    return total_loss


# Loss function for WGAN - Eart mover's distance -> Wasserstein loss
def critic_loss_function(real_prediction, fake_prediction, critic):
    real_loss = -mean(real_prediction)
    fake_loss = mean(fake_prediction)

    total_loss = real_loss + fake_loss
    return total_loss


# Discriminator architecture - 3 Conv2D layers, 3 LeakyReLU layers, 1 Flatten, 1 Dropout and 1 Dense
class Discriminator(keras.Model):

    # Initialize discriminator
    def __init__(self, img_size=32):
        super(Discriminator, self).__init__(name='discriminator')

        self.conv2d_1 = keras.layers.Conv2D(2*img_size, (3, 3),
                                            input_shape=(img_size, img_size, 3), padding='same')
        self.leaky_1 = keras.layers.LeakyReLU(alpha=0.2)

        self.conv2d_2 = keras.layers.Conv2D(4*img_size, (3, 3), strides=(2, 2), padding='same')
        self.leaky_2 = keras.layers.LeakyReLU(alpha=0.2)

        self.conv2d_3 = keras.layers.Conv2D(4 * img_size, (3, 3), strides=(2, 2), padding='same')
        self.leaky_3 = keras.layers.LeakyReLU(alpha=0.2)

        self.flatten = keras.layers.Flatten()
        self.dropout = keras.layers.Dropout(0.4)
        self.dense = keras.layers.Dense(1, activation='sigmoid')

    # The output of the previous layer is the input to the next one
    def call(self, inputs):
        x = self.conv2d_1(inputs)
        x = self.leaky_1(x)

        x = self.conv2d_2(x)
        x = self.leaky_2(x)

        x = self.conv2d_3(x)
        x = self.leaky_3(x)

        x = self.flatten(x)
        x = self.dropout(x)
        x = self.dense(x)

        return x


class ConditionalDiscriminator(keras.Model):
    def __init__(self, img_size=32, num_classes=16):
        super(ConditionalDiscriminator, self).__init__(name='conditional_discriminator')

        self.embedding = keras.layers.Embedding(num_classes, 50)
        self.scaling = keras.layers.Dense(img_size*img_size*3)
        self.reshape = keras.layers.Reshape((img_size, img_size, 3))

        self.merge = keras.layers.Concatenate()

        self.conv2d_1 = keras.layers.Conv2D(2 * img_size, (3, 3),
                                            input_shape=(img_size, img_size, 3), padding='same')
        self.leaky_1 = keras.layers.LeakyReLU(alpha=0.2)

        self.conv2d_2 = keras.layers.Conv2D(4 * img_size, (3, 3), strides=(2, 2), padding='same')
        self.leaky_2 = keras.layers.LeakyReLU(alpha=0.2)

        self.conv2d_3 = keras.layers.Conv2D(4 * img_size, (3, 3), strides=(2, 2), padding='same')
        self.leaky_3 = keras.layers.LeakyReLU(alpha=0.2)

        self.flatten = keras.layers.Flatten()
        self.dropout = keras.layers.Dropout(0.4)
        self.dense = keras.layers.Dense(1, activation='sigmoid')

    def call(self, input_image, input_label):
        x = self.embedding(input_label)
        x = self.scaling(x)
        x = self.reshape(x)

        x = self.merge([input_image, x])

        x = self.conv2d_1(x)
        x = self.leaky_1(x)

        x = self.conv2d_2(x)
        x = self.leaky_2(x)

        x = self.conv2d_3(x)
        x = self.leaky_3(x)

        x = self.flatten(x)
        x = self.dropout(x)
        x = self.dense(x)

        return x


"""
    The same architecture as the Discriminator, but with weight clipping (kernel_constraint) and no activation function
    for the last layer. We call Critic the Discriminator used in WGAN, because it does not actually classify images as real
    and fake, but outputs a real number and based on that number the Generator updates it's weights
"""


class Critic(keras.Model):

    # Initialize Critic
    def __init__(self, img_size=64):
        super(Critic, self).__init__(name='critic')

        self.conv2d_1 = keras.layers.Conv2D(2 * img_size, (3, 3),
                                            input_shape=(img_size, img_size, 3), padding='same',
                                            kernel_constraint=constraint)
        self.leaky_1 = keras.layers.LeakyReLU(alpha=0.2)

        self.conv2d_2 = keras.layers.Conv2D(4 * img_size, (3, 3), strides=(2, 2), padding='same',
                                            kernel_constraint=constraint)
        self.leaky_2 = keras.layers.LeakyReLU(alpha=0.2)

        self.conv2d_3 = keras.layers.Conv2D(4 * img_size, (3, 3), strides=(2, 2), padding='same',
                                            kernel_constraint=constraint)
        self.leaky_3 = keras.layers.LeakyReLU(alpha=0.2)

        self.flatten = keras.layers.Flatten()
        self.dropout = keras.layers.Dropout(0.4)
        self.dense = keras.layers.Dense(1)

    # The output of the previous layer is the input to the next one
    def call(self, inputs):
        x = self.conv2d_1(inputs)
        x = self.leaky_1(x)

        x = self.conv2d_2(x)
        x = self.leaky_2(x)

        x = self.conv2d_3(x)
        x = self.leaky_3(x)

        x = self.flatten(x)
        x = self.dropout(x)
        x = self.dense(x)

        return x
