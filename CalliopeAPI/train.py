from discriminator import Discriminator, discriminator_loss_function, bin_cross_entropy
from generator import Generator, generator_loss_function, generate_noise
from optimizers import define_optimizers
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

# Training step - feed n(batch_size) real images to the GAN
@tf.function()
def training_step(generator: Generator, discriminator: Discriminator, generator_optimizer, discriminator_optimizer, images: np.ndarray, k: int = 1, batch_size=1):
    for i in range(k):
        with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
            # Generate n(batch_size) fake images
            noise = generate_noise(batch_size, 100)
            generated_images = generator(noise, training=True)

            # Get the predictions of the discriminator
            real_prediction = discriminator(images, training=True)
            fake_prediction = discriminator(generated_images, training=True)

            # Calculate the losses
            generator_loss = generator_loss_function(fake_prediction)
            discriminator_loss = discriminator_loss_function(real_prediction, fake_prediction)

            # Optimize the discriminator
            gradients_of_discriminator = discriminator_tape.gradient(discriminator_loss, discriminator.trainable_variables)
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

            # Optimize the generator
            gradients_of_generator = generator_tape.gradient(generator_loss, generator.trainable_variables)
            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))


# Train the GAN on all images of the dataset
def train(generator: Generator, discriminator: Discriminator, generator_optimizer, discriminator_optimizer, training_images, epochs, batch_size):
    for epoch in range(epochs):
        for batch in training_images:
            training_step(generator, discriminator, generator_optimizer, discriminator_optimizer, batch, k=1, batch_size=batch_size)

        # On every 20 epochs generate one image and show it
        if epoch % 20 == 0:
            fake_image = generator(generate_noise(batch_size=1, random_noise_size=100), training=False)
            plt.imshow(fake_image[0])
            plt.show()


def main():
    # Define parameters
    img_size = 64
    epochs = 500
    batch_size = 100

    # Create instances of the Generator, Discriminator and the optimizers
    generator = Generator(img_size=img_size)
    discriminator = Discriminator(img_size=img_size)
    generator_optimizer, discriminator_optimizer = define_optimizers()

    # Load the dataset and normalize it
    images = np.load('dataset/images_dataset.npy', allow_pickle=True)
    images = (images - 127.5) / 127.5
    images = images.astype('float32')
    images = images[:400]

    # Slice the dataset into batches of size 100
    images = tf.data.Dataset.from_tensor_slices(images).batch(batch_size)

    # Train the GAN on the dataset for 100 epochs
    train(generator, discriminator, generator_optimizer, discriminator_optimizer, images, epochs, batch_size)


if __name__ == '__main__':
    main()
