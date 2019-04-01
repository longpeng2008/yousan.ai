import sys
import cv2
import numpy as np
import tensorflow as tf
import keras.backend as K

from net import simpleconv3
from PIL import Image
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import preprocess_input

image_size = (48, 48)
batch_shape = (1, ) + image_size + (3, )
model_path = sys.argv[1]
# model_path = './models/model.h5'

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
K.set_session(session)

model = simpleconv3()
model.load_weights(model_path, by_name=True)
model.summary()

image_path = sys.argv[2]
# image_path = '../../../../datas/head/train/0/1left.jpg'
img = Image.open(image_path)
img = img_to_array(img)
img = cv2.resize(img, image_size)
img = np.expand_dims(img, axis=0)
img = preprocess_input(img)

result = model.predict(img, batch_size=1)
print(result)
