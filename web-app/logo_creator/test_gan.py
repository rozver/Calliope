
import tensorflow as tf
import os
from matplotlib import pyplot as plt
from generator import Generator, generate_noise
import cv2
import numpy as np


def generate():
    generator = Generator(img_size=32)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    generator.load_weights('../CalliopeAPI/generator_weights/generator')

    img = generator(generate_noise(1, 100))[0]
    print(img)

    cv2.imwrite('static/output.png', np.array(img * 255))


if __name__ == '__main__':
    generate()
