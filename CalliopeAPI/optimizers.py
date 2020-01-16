from tensorflow import keras


# Define two identical, but distinct optimizers
def define_optimizers():
    generator_optimizer = keras.optimizers.Adam(1e-4)
    discriminator_optimizer = keras.optimizers.Adam(1e-4)
    return generator_optimizer, discriminator_optimizer
