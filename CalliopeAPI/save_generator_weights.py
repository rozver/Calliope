import tensorflow as tf
from generator import Generator, generate_noise
from discriminator import Discriminator
from optimizers import define_dcgan_optimizers
import os


def restore_checkpoint(checkpoint, checkpoint_dir):
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    print('Checkpoint restored')


def save_weights():
    generator = Generator(img_size=32)
    discriminator = Discriminator(img_size=32)
    generator_optimizer, discriminator_optimizer = define_dcgan_optimizers()

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    checkpoint = tf.train.Checkpoint(generator=generator,
                                     discriminator=discriminator,
                                     generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer)

    restore_checkpoint(checkpoint, 'models/checkpoints')

    generator.build(input_shape=(None, 100))
    test_prediction = generator.predict(generate_noise(1, 100))
    print(test_prediction)

    generator.save_weights('generator_weights/generator')


if __name__ == '__main__':
    save_weights()