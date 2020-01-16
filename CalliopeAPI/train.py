from discriminator import Discriminator, discriminator_loss_function
from generator import Generator, generator_loss_function, generate_noise
from optimizers import define_optimizers
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import os
import pickle


# Create a checkpoint
def create_checkpoint(generator: Generator, discriminator: Discriminator, generator_optimizer, discriminator_optimizer):

    checkpoint = tf.train.Checkpoint(generator=generator,
                                     discriminator=discriminator,
                                     generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer)
    return checkpoint


def save_checkpoint(checkpoint, checkpoint_dir):
    checkpoint_file_prefix = os.path.join(checkpoint_dir, 'ckpt')
    checkpoint.save(file_prefix=checkpoint_file_prefix)


def restore_checkpoint(checkpoint, checkpoint_dir):
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# Training step - feed n(batch_size) real images to the GAN
@tf.function()
def training_step(generator: Generator, discriminator: Discriminator, generator_optimizer,
                  discriminator_optimizer, images: np.ndarray, k: int = 1, batch_size=1):
    for i in range(k):
        """ 
        # Discriminator training
        for _ in range(5):
            with tf.GradientTape() as discriminator_tape:
                noise = generate_noise(batch_size, 100)
                generated_images = generator(noise, training=False)
                real_prediction = discriminator(images, training=True)
                fake_prediction = discriminator(generated_images, training=True)
                discriminator_loss = discriminator_loss_function(real_prediction, fake_prediction)
                gradients_of_discriminator = discriminator_tape.gradient(discriminator_loss,
                                                                         discriminator.trainable_variables)
                discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator,
                                                        discriminator.trainable_variables))
        """

        # Generator training
        with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
            noise = generate_noise(batch_size, 100)
            generated_images = generator(noise, training=True)

            # Get the predictions of the discriminator
            real_prediction = discriminator(images, training=True)
            fake_prediction = discriminator(generated_images, training=True)

            # Calculate the losses
            generator_loss = generator_loss_function(fake_prediction)
            discriminator_loss = discriminator_loss_function(real_prediction, fake_prediction)
        
        print('Generator loss: ' + str(generator_loss))
        print('Discriminator loss: ' + str(discriminator_loss))          
  
            # Optimize the discriminator
        gradients_of_discriminator = discriminator_tape.gradient(discriminator_loss,
                                                                     discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator,
                                                        discriminator.trainable_variables))

            # Optimize the generator
        gradients_of_generator = generator_tape.gradient(generator_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
  
        print('Trained on another batch')

# Train the GAN on all images of the dataset
def train(generator: Generator, discriminator: Discriminator, generator_optimizer,
          discriminator_optimizer, training_images, epochs, batch_size):
    for epoch in range(1, epochs+1):
        print('Epoch: ' + str(epoch) + '/' + str(epochs))
        for batch in training_images:
            training_step(generator, discriminator, generator_optimizer, discriminator_optimizer, batch, k=1,
                          batch_size=batch_size)

        # On every 20 epochs generate one image save it
        if epoch % 2 == 0:
            fake_image = generator(generate_noise(batch_size=1, random_noise_size=100), training=False)
            # print('Generator loss: ' + str(generator_loss_function(discriminator(fake_image, training=False))))
            plt.imshow(fake_image[0])
            plt.savefig('{}/{}.png'.format('generated_images', epoch))


def main():
    # Define parameters
    img_size = 32
    epochs = 500
    batch_size = 32
    checkpoint_dir = 'models/checkpoints/'

    # Create instances of the Generator, Discriminator and the optimizers
    generator = Generator(img_size=img_size)
    discriminator = Discriminator(img_size=img_size)
    generator_optimizer, discriminator_optimizer = define_optimizers()

    """
    # Restore checkpoint
    print('Restoring checkpoint...')
    old_checkpoint = create_checkpoint(generator, discriminator, generator_optimizer, discriminator_optimizer)
    restore_checkpoint(old_checkpoint, checkpoint_dir)
    print('Checkpoint restored')
    """

    # Load the dataset and normalize it
    print('Loading dataset...')
    images = np.load('icon_dataset_2_test.npy', allow_pickle=True)
    images = images[:int(len(images)/2)]
    # images = images / 255.0
    print('Finished')

    # Slice the dataset into batches of size batch_size
    print('Splitting the dataset into batches...')
    images = tf.data.Dataset.from_tensor_slices(images).batch(batch_size)
    print('Finished')

    # Train the GAN on the dataset
    print('Starting training...')
    train(generator, discriminator, generator_optimizer, discriminator_optimizer, images, epochs, batch_size)
    print('Training finished')

    # Create a checkpoint
    print('Creating checkpoint...')
    new_checkpoint = create_checkpoint(generator, discriminator, generator_optimizer, discriminator_optimizer)
    save_checkpoint(new_checkpoint, checkpoint_dir)
    print('Checkpoint created')


if __name__ == '__main__':
    main()
