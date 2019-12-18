from discriminator import Discriminator, discriminator_loss_function, bin_cross_entropy
from generator import Generator, generator_loss_function
from optimizers import generator_optimizer, discriminator_optimizer
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

# Define image size and the testing generator and discriminator
IMG_SIZE = 128
test_generator = Generator()
test_discriminator = Discriminator()

# Training step - feed n(batch_size) images to the GAN
@tf.function()
def training_step(generator: Generator, discriminator: Discriminator, images: np.ndarray, k: int = 1, batch_size=1):
    for i in range(k):
        with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
            # Generate random noise
            noise = generator.generate_noise(batch_size, 100)
            # Generate images
            generated_images = generator(noise)

            # Get the predictions of the discriminator
            discriminator_real_prediction = discriminator(images)
            discriminator_fake_prediction = discriminator(generated_images)

            # Calculate discriminator loss and optimize it
            discriminator_loss = discriminator_loss_function(discriminator_real_prediction, discriminator_fake_prediction)
            gradients_of_discriminator = discriminator_tape.gradient(discriminator_loss, discriminator.trainable_variables)
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

            # Calculate generator loss and optimize it
            generator_loss = generator_loss_function(generated_images)
            gradients_of_generator = generator_tape.gradient(generator_loss, generator.trainable_variables)
            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))


# Train the GAN on all images of the dataset
def train(training_images, epochs):
    for epoch in range(epochs):
        training_step(test_generator, test_discriminator, training_images, k=1, batch_size=1)

        # On every 20 epochs show the result
        if epoch % 20 == 0:
            fake_image = test_generator(test_generator.generate_noise(batch_size=1, random_noise_size=100))
            plt.imshow(fake_image[0])
            plt.show()


# Load the dataset and normalize it
images = np.load('dataset/images_dataset.npy', allow_pickle=True)
images = np.array(images).reshape(-1, 128, 128, 3)

images = (images - 127.5) / 127.5
images = images[:100]  # Running on only 100 out of 4096 images - otherwise it would be too heavy for the CPU

# Train the GAN on the dataset for 100 epochs
train(images, 100)

