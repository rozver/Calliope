import tensorflow as tf
import os
from matplotlib import pyplot as plt
from generator import Generator, generate_noise
import cv2
import numpy as np
import sys


def generate():
    username = str(sys.argv[1])
    generator = Generator(img_size=32)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    generator.load_weights('../CalliopeAPI/generator_weights/generator')

    img = generator(generate_noise(1, 100))[0]
    plt.imshow(img)
    plt.axis('off')
    plt.savefig('media/temporary_logos/' + username + '_output.png', bbox_inches='tight')
    # cv2.imwrite('media/temporary_logos/' + username + '_output.png', np.array(img * 255))


if __name__ == '__main__':
    generate()
