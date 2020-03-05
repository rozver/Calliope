import tensorflow as tf
from generator import Generator, generate_noise
from discriminator import Discriminator
from optimizers import define_lsgan_optimizers
import os


# Restore the latest checkpoint
def restore_checkpoint(checkpoint, checkpoint_dir):
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    print('Checkpoint restored')


# Function for saving the weights of the Generator inside the specified folder - current implementation is for LSGAN
def save_weights():
    # Define Generator, Discriminator and their corresponding optimizers
    generator = Generator(img_size=32)
    discriminator = Discriminator(img_size=32)
    generator_optimizer, discriminator_optimizer = define_lsgan_optimizers()

    # Set Tensorflow to run on CPU - that is not training
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # Restore checkpoint
    checkpoint = tf.train.Checkpoint(generator=generator,
                                     discriminator=discriminator,
                                     generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer)

    restore_checkpoint(checkpoint, 'models/checkpoints')

    # Build the Generator and specify its input shape
    generator.build(input_shape=(None, 100))

    # Make a test prediction to see whether it is working
    test_prediction = generator.predict(generate_noise(1, 100))
    print(test_prediction)

    # Save weights
    generator.save_weights('generator_weights/generator')


if __name__ == '__main__':
    save_weights()