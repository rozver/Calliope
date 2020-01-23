from tensorflow import keras


# Define two identical, but distinct optimizers

# DCGAN
def define_dcgan_optimizers():
    generator_optimizer = keras.optimizers.Adam(1e-4)
    discriminator_optimizer = keras.optimizers.Adam(1e-4)
    return generator_optimizer, discriminator_optimizer


# WGAN
def define_wgan_optimizers():
    generator_optimzer = keras.optimizers.RMSProp(lr=0.00005)
    discriminator_optimizer = keras.optimizers.RMSProp(lr=0.00005)
    return generator_optimzer, discriminator_optimizer
