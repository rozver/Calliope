from tensorflow import keras


# Define two identical, but distinct optimizers
def define_optimizers():
    generator_optimizer = keras.optimizers.RMSprop()
    discriminator_optimizer = keras.optimizers.RMSprop()
    return generator_optimizer, discriminator_optimizer
