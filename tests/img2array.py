import tensorflow as tf 
from keras.preprocessing import image
import numpy as np

def prepare_image(file):
    img_path = ''
    img = tf.keras.preprocessing.image.load_img(img_path + file, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet_v2.preprocess_input(img_array_expanded_dims)

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


img = "tests1.jpg"

img = prepare_image(img)

model = tf.keras.models.load_model('models/test.h5')

pred = model.predict(img)
print(sigmoid(pred))
print(np.argmax(pred))
