from tensorflow import keras

# Define two identical, but distinct optimizers
generator_optimizer = keras.optimizers.RMSprop()
discriminator_optimizer = keras.optimizers.RMSprop()

