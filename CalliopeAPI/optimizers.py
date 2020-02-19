from tensorflow import keras


# Define two identical, but distinct optimizers

# DCGAN
def define_dcgan_optimizers():
    generator_optimizer = keras.optimizers.Adam(1e-4)
    discriminator_optimizer = keras.optimizers.Adam(1e-4)
    return generator_optimizer, discriminator_optimizer


# WGAN - currently experimenting with Adam optimizer for Generator to compare results
def define_wgan_optimizers():
    generator_optimzer = keras.optimizers.Adam(0.0001, beta_1=0.5)
    critic_optimizer = keras.optimizers.RMSprop(lr=0.00005)
    return generator_optimzer, critic_optimizer
