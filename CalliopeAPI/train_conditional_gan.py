from discriminator import ConditionalDiscriminator, discriminator_lsgan_loss_function
from generator import ConditionalGenerator, generate_noise, generator_lsgan_loss_function
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from optimizers import define_lsgan_optimizers
import os
import random


def create_checkpoint_clsgan(generator: ConditionalGenerator, discriminator: ConditionalDiscriminator,
                             generator_optimizer, discriminator_optimizer):

    checkpoint = tf.train.Checkpoint(generator=generator,
                                     discriminator=discriminator,
                                     generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer)
    return checkpoint


# Restore the latest checkpoint
def restore_checkpoint(checkpoint, checkpoint_dir):
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


# Save the current checkpoint checkpoint
def save_checkpoint(checkpoint, checkpoint_dir):
    checkpoint_file_prefix = os.path.join(checkpoint_dir, 'ckpt')
    checkpoint.save(file_prefix=checkpoint_file_prefix)


def generate_and_save_sample_image(generator: ConditionalGenerator, labels_batch, batch_size, architecture_type, epoch):
    noise = generate_noise(batch_size, 100)
    generated_image = generator(noise, labels_batch)[0]
    plt.imshow(generated_image)
    plt.axis('off')
    plt.savefig('{}/{}/{}.png'.format('generated_images/', architecture_type, epoch), bbox_inches='tight')


@tf.function
def training_step_clsgan(generator: ConditionalGenerator, discriminator: ConditionalDiscriminator, generator_optimizer,
                         discriminator_optimizer, images_batch: np.ndarray, labels_batch: np.ndarray, batch_size,
                         generator_training_rate, discriminator_training_rate):
    for _ in range(generator_training_rate):
        with tf.GradientTape() as generator_tape:
            noise = generate_noise(batch_size, 100)
            generated_images = generator(noise, labels_batch, training=True)

            # Get the predictions of the Discriminator
            fake_prediction = discriminator(generated_images, labels_batch, training=True)

            # Calculate the losses
            generator_loss = generator_lsgan_loss_function(fake_prediction)

            gradients_of_generator = generator_tape.gradient(generator_loss, generator.trainable_variables)
            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    for _ in range(discriminator_training_rate):
        with tf.GradientTape() as discriminator_tape:
            noise = generate_noise(batch_size, 100)
            generated_images = generator(noise, labels_batch, training=False)
            real_prediction = discriminator(images_batch, labels_batch, training=False)
            fake_prediction = discriminator(generated_images, labels_batch, training=True)
            discriminator_loss = discriminator_lsgan_loss_function(real_prediction, fake_prediction)

            # Optimize the Discriminator
            gradients_of_discriminator = discriminator_tape.gradient(discriminator_loss,
                                                                     discriminator.trainable_variables)
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator,
                                                        discriminator.trainable_variables))


def train_clsgan(generator: ConditionalGenerator, discriminator: ConditionalDiscriminator, generator_optimizer,
                 discriminator_optimizer, training_images, training_labels, epochs, batch_size, generator_training_rate,
                 discriminator_training_rate):
    for epoch in range(epochs):
        for images_batch, labels_batch in zip(training_images, training_labels):
            training_step_clsgan(generator, discriminator, generator_optimizer, discriminator_optimizer, images_batch,
                                 labels_batch, batch_size, generator_training_rate, discriminator_training_rate)
        if epoch % 2 == 0:
            generate_and_save_sample_image(generator, labels_batch, batch_size, 'clsgan', epoch)


def main():
    generator_training_rate = 1
    discriminator_training_rate = 1
    checkpoint_dir = 'models/checkpoints/clsgan'
    epochs = 30
    batch_size = 32

    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

    generator = ConditionalGenerator()
    discriminator = ConditionalDiscriminator()

    generator_optimizer, discriminator_optimizer = define_lsgan_optimizers()

    old_checkpoint = create_checkpoint_clsgan(generator, discriminator,
                                              generator_optimizer, discriminator_optimizer)
    restore_checkpoint(old_checkpoint, checkpoint_dir)

    images = np.load('dataset/LLD_icon_numpy/dataset1.npy', allow_pickle=True)
    labels = np.load('dataset/LLD_icon_numpy/dataset1_labels.npy', allow_pickle=True)

    images = tf.data.Dataset.from_tensor_slices(images).batch(batch_size)
    labels = tf.data.Dataset.from_tensor_slices(labels).batch(batch_size)

    train_clsgan(generator, discriminator, generator_optimizer, discriminator_optimizer,
                 images, labels, epochs, batch_size, generator_training_rate, discriminator_training_rate)

    # Create a checkpoint
    print('Creating checkpoint...')
    new_checkpoint = create_checkpoint_clsgan(generator, discriminator,
                                              generator_optimizer, discriminator_optimizer)
    save_checkpoint(new_checkpoint, checkpoint_dir)
    print('Checkpoint created')


if __name__ == '__main__':
    main()
