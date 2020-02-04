from django.shortcuts import render
from django.http import HttpResponseRedirect, HttpResponse
from .generator import Generator, generate_noise
from .discriminator import Discriminator
from .optimizers import define_dcgan_optimizers
from .complementary_colors import complement
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import cv2

# Create your views here.


def index(request):
    return render(request, 'index.html')


def about(request):
    return render(request, 'about-us.html')


def info(request):
    return render(request, 'info.html')


def contact_us(request):
    return render(request, 'contact-us.html')


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

    restore_checkpoint(checkpoint, '/home/rozver/Documents/Calliope/interface/logo_creator/checkpoints')

    img = generator(generate_noise(1, 100))
    img = img[0]

    cv2.imwrite('static/output.png', np.array(img*255))
    return render(request, 'index.html')


def complement_image(request):
    complement('/home/rozver/Documents/Calliope/interface/static/output.png')
    return render(request, 'index.html')
