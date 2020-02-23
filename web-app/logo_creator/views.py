from django.shortcuts import render
from django.http import HttpResponseRedirect, HttpResponse
from .generator import Generator, generate_noise
from .discriminator import Discriminator
from .optimizers import define_dcgan_optimizers
from .complementary_colors import complement
import tensorflow as tf
import numpy as np
import cv2


def home(request):
    return render(request, 'home.html')


# Create your views here.
def restore_checkpoint(checkpoint, checkpoint_dir):
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    print('Checkpoint restored')


def generate_logo(request):
    generator = Generator(img_size=32)
    discriminator = Discriminator(img_size=32)
    generator_optimizer, discriminator_optimizer = define_dcgan_optimizers()

    checkpoint = tf.train.Checkpoint(generator=generator,
                                     discriminator=discriminator,
                                     generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer)

    restore_checkpoint(checkpoint, '../../CalliopeAPI/models/checkpoints')

    img = generator(generate_noise(1, 100))
    img = img[0]

    cv2.imwrite('static/output.png', np.array(img*255))
    return render(request, 'home.html')


def complement_image(request):
    complement('../static/output.png')
    return render(request, 'home.html')
