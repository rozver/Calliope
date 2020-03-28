from tensorflow import keras


def vanilla_autoencoder_loss_function(original, reconstructed):
    original, reconstructed = keras.backend.flatten(original), keras.backend.flatten(reconstructed)
    loss = keras.losses.MSE(original, reconstructed)
    return loss


class Encoder(keras.layers.Layer):
    def __init__(self, original_size, intermediate_size, channels):
        super(Encoder, self).__init__(name='encoder')
        self.flatten = keras.layers.Flatten(input_shape=(original_size, original_size, channels))
        self.hidden_layer_1 = keras.layers.Dense(
            intermediate_size,
            activation='relu'
        )
        self.hidden_layer_2 = keras.layers.Dense(
            intermediate_size,
            activation='relu'
        )
        self.hidden_layer_3 = keras.layers.Dense(
            intermediate_size,
            activation='relu'
        )

        self.output_layer = keras.layers.Flatten()

    def call(self, input_features):
        x = self.flatten(input_features)
        x = self.hidden_layer_1(x)
        x = self.hidden_layer_2(x)
        x = self.hidden_layer_3(x)
        x = self.output_layer(x)
        return x


class Decoder(keras.layers.Layer):
    def __init__(self, original_size, intermediate_size, channels):
        super(Decoder, self).__init__(name='decoder')
        self.input_layer = keras.layers.Dense(intermediate_size,
                                              input_shape=(intermediate_size, ),
                                              activation='relu')
        self.hidden_layer_1 = keras.layers.Dense(
            intermediate_size,
            activation='relu'
        )

        self.hidden_layer_2 = keras.layers.Dense(
            intermediate_size,
            activation='relu'
        )

        self.output_layer = keras.layers.Dense(
            original_size*original_size*channels,
            activation='sigmoid'
        )

    def call(self, code):
        x = self.input_layer(code)
        x = self.hidden_layer_1(x)
        x = self.hidden_layer_2(x)
        x = self.output_layer(x)
        return x


class Autoencoder(keras.Model):
    def __init__(self, original_size, intermediate_size, channels):
        super(Autoencoder, self).__init__(name='autoencoder')
        self.encoder = Encoder(original_size, intermediate_size, channels)
        self.decoder = Decoder(original_size, intermediate_size, channels)

    def call(self, input_features):
        code = self.encoder(input_features)
        reconstructed = self.decoder(code)
        return code, reconstructed
