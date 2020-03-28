import tensorflow as tf
from tensorflow import keras
from vanilla_autoencoder import Autoencoder as VanillaAutoencoder
from convolutional_autoencoder import Autoencoder as ConvolutionalAutoencoder
from vanilla_autoencoder import vanilla_autoencoder_loss_function
from convolutional_autoencoder import convolutional_autoencoder_loss_function
from matplotlib import pyplot as plt
import random
import numpy as np
import os


# Vanilla autoencoder training step
@tf.function
def training_step_vanilla_autoencoder(autoencoder: VanillaAutoencoder, optimizer, batch):
    with tf.GradientTape() as tape:
        code, reconstructed = autoencoder(batch)
        loss = vanilla_autoencoder_loss_function(batch, reconstructed)
        gradients = tape.gradient(loss, autoencoder.trainable_variables)
        optimizer.apply_gradients(zip(gradients, autoencoder.trainable_variables))


# Convolutional autoencoder training step
@tf.function
def training_step_convolutional_autoencoder(autoencoder: ConvolutionalAutoencoder, optimizer, batch):
    with tf.GradientTape() as tape:
        code, reconstructed = autoencoder(batch)
        loss = convolutional_autoencoder_loss_function(batch, reconstructed)
        gradients = tape.gradient(loss, autoencoder.trainable_variables)
        optimizer.apply_gradients(zip(gradients, autoencoder.trainable_variables))


# Training function for Vanilla autoencoder
def train_vanilla_autoencoder(autoencoder: VanillaAutoencoder, optimizer, images, epochs, original_size, channels):
    for epoch in range(epochs):
        print('Epoch: ' + str(epoch) + '/' + str(epochs))
        for batch in images:
            training_step_vanilla_autoencoder(autoencoder, optimizer, batch)

            # Reconstruct an image and save it on every 5 epochs
            if (epoch+1) % 5 == 0:
                codes, reconstructed_images = autoencoder(batch)
                reconstructed_image_random_index = random.randint(1, len(reconstructed_images) - 1)

                # If it is a grayscale image (it has one channel), it must be reshaped without specifying channel
                if channels == 1:
                    reconstructed_images = tf.reshape(
                        reconstructed_images[reconstructed_image_random_index], (original_size, original_size)
                    )
                else:
                    reconstructed_images = tf.reshape(
                        reconstructed_images[reconstructed_image_random_index], (original_size, original_size, channels)
                    )

                plt.imshow(reconstructed_images, cmap='gray')
                plt.savefig('generated_images/vanilla_autoencoder/' + str(epoch+1) + '.png')


# Training function for Convolutional autoencoder
def train_convolutional_autoencoder(autoencoder: ConvolutionalAutoencoder, optimizer, images, epochs):
    for epoch in range(epochs):
        print('Epoch: ' + str(epoch+1) + '/' + str(epochs))
        for batch in images:
            training_step_convolutional_autoencoder(autoencoder, optimizer, batch)

        # Reconstruct an image and save it on every 5 epochs
            if (epoch+1) % 1 == 0:
                codes, reconstructed_images = autoencoder(batch, training=False)
                # print(reconstructed_images)
                print(codes)
                reconstructed_image_random_index = random.randint(1, len(reconstructed_images) - 1)

                plt.imshow(reconstructed_images[reconstructed_image_random_index])
                plt.savefig('generated_images/convolutional_autoencoder/' + str(epoch+1) + '.png')


def main():
    # Define parameters
    original_size = 32
    intermediate_size = 128
    channels = 3
    epochs = 30
    batch_size = 16

    # Allow memory growth for GPU
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

    # Autoencoder architecture selection
    print('Select autoencoder architecture: 1 (Vanilla) or 2 (Convolutional)')
    mode = int(input())
    print('Architecture selected')

    print('Loading dataset...')
    images = np.load('dataset/LLD_icon_numpy/dataset1.npy', allow_pickle=True)
    print('Finished')

    # Preprocess dataset - slice, batch and shuffle
    print('Preprocessing dataset..')
    images = tf.data.Dataset.from_tensor_slices(images).batch(batch_size).shuffle(len(images))
    print('Finished')

    if mode == 1:
        # Defining Vanilla autoencoder and its optimizer
        autoencoder = VanillaAutoencoder(original_size, intermediate_size, channels)
        optimizer = keras.optimizers.Adam(0.02)

        # Start training
        print('Starting training...')
        train_vanilla_autoencoder(autoencoder, optimizer, images, epochs, original_size, channels)
        print('Finished')

    elif mode == 2:
        # Defining Convolutional autoencoder and its optimizer
        autoencoder = ConvolutionalAutoencoder(original_size, intermediate_size)
        optimizer = keras.optimizers.Adam(0.02)

        # Start training
        print('Starting training...')
        train_convolutional_autoencoder(autoencoder, optimizer, images, epochs)
        print('Finished')

    else:
        print('Invalid architecture selected! Select 1 (Vanilla) or 2 (Convolutional)!')


if __name__ == '__main__':
    main()
