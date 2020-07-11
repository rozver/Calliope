from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras import Input
from skimage import transform as st
import cv2
import tensorflow as tf

def get_predictions(images, model):
    predictions = model.predict(images)
    return predictions


images = np.load('dataset/LLD_icon_numpy/dataset1.npy')
model = InceptionV3(include_top=True, weights='imagenet')

start_index = 0
end_index = 2000
count = 0

while count<50:
    images_batch = images[start_index:end_index]
    image_list = []
    for img in images_batch:
        img = cv2.resize(img, dsize=(299, 299), interpolation=cv2.INTER_CUBIC)
        image_list.append(img)

    images_list = np.array(image_list)

    predictions = model.predict(images_list)

    decoded_predictions = decode_predictions(predictions, top=3)

    decoded_predictions = np.array(decoded_predictions)
    
    np.save('predictions/decoded_predictions_'+str(count)+'.npy', decoded_predictions)
    start_index = start_index + 2000
    end_index = end_index + 2000
    count = count + 1

