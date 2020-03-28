from tensorflow import keras


def convolutional_autoencoder_loss_function(original, reconstructed):
    original, reconstructed = keras.backend.flatten(original), keras.backend.flatten(reconstructed)
    loss = keras.losses.MSE(original, reconstructed)
    return loss


class Encoder(keras.layers.Layer):
    def __init__(self, original_size):
        super(Encoder, self).__init__(name='encoder')
        self.conv2d_1 = keras.layers.Conv2D(16, (3, 3), padding='same', input_shape=(original_size, original_size, 3),
                                            activation='relu')
        self.max_pooling_1 = keras.layers.MaxPooling2D((2, 2), padding='same')

        self.conv2d_2 = keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu')
        self.max_pooling_2 = keras.layers.MaxPooling2D((2, 2), padding='same')

        self.conv2d_3 = keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu')
        self.max_pooling_3 = keras.layers.MaxPooling2D((2, 2), padding='same')

        self.ouput_layer = keras.layers.Flatten()

    def call(self, input_features):
        x = self.conv2d_1(input_features)
        x = self.max_pooling_1(x)

        x = self.conv2d_2(x)
        x = self.max_pooling_2(x)

        x = self.conv2d_3(x)
        x = self.max_pooling_3(x)

        x = self.ouput_layer(x)
        return x


class Decoder(keras.layers.Layer):
    def __init__(self, intermediate_size):
        img_size = 32
        starting_image_size = int(img_size/4)
        super(Decoder, self).__init__(name='decoder')
        self.input_layer = keras.layers.Dense(starting_image_size*starting_image_size*256,
                                              input_shape=(intermediate_size,))
        self.leaky_1 = keras.layers.LeakyReLU(alpha=0.2)
        self.reshape = keras.layers.Reshape((starting_image_size, starting_image_size, 256))

        self.conv2d_transpose_1 = keras.layers.Conv2DTranspose(starting_image_size * 32, (4, 4), strides=(2, 2),
                                                               padding='same')
        self.leaky_2 = keras.layers.LeakyReLU(alpha=0.2)

        self.conv2d_transpose_2 = keras.layers.Conv2DTranspose(img_size * 32, (4, 4), strides=(2, 2),
                                                               padding='same')

        self.leaky_3 = keras.layers.LeakyReLU(alpha=0.2)

        self.output_layer = keras.layers.Conv2DTranspose(3, (3, 3),
                                                         padding='same', activation='tanh')

    def call(self, code):
        x = self.input_layer(code)
        x = self.leaky_1(x)
        x = self.reshape(x)

        x = self.conv2d_transpose_1(x)
        x = self.leaky_2(x)

        x = self.conv2d_transpose_2(x)
        x = self.leaky_3(x)

        x = self.output_layer(x)
        return x


class Autoencoder(keras.Model):
    def __init__(self, original_size, intermediate_size):
        super(Autoencoder, self).__init__(name='autoencoder')
        self.encoder = Encoder(original_size)
        self.decoder = Decoder(intermediate_size)

    def call(self, input_features):
        code = self.encoder(input_features)
        reconstructed = self.decoder(code)
        return code, reconstructed
