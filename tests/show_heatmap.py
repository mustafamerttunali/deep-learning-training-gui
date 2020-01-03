### Not working.

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import cv2

tf.compat.v1.disable_eager_execution()
def prepare_image(img):
        img = tf.keras.preprocessing.image.load_img(img, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array_expanded_dims = np.expand_dims(img_array, axis=0)
        return tf.keras.applications.mobilenet_v2.preprocess_input(img_array_expanded_dims)


img2array = prepare_image('static/test2.jpg')
model = tf.keras.models.load_model('models/MobileNetV2_Flowers.h5')

pred = model.predict(img2array)

ind=np.argmax(pred[0])
vector=model.output[:,ind]
last_conv=model.get_layer("dense")
grads=tf.keras.backend.gradients(vector,last_conv.output)[0]
pooled_grad=tf.keras.backend.mean(grads)
iterate=tf.keras.backend.function([model.input],[pooled_grad,last_conv.output[0]])
pooled_grad_value,conv_layer_value=iterate(img2array)

heatmap=np.mean(pooled_grad_value)

heatmap=np.maximum(heatmap,0)
heatmap /= np.max(heatmap)
heatmap=cv2.resize(heatmap,(img2array.shape[1],img2array.shape[0]))
heatmap=np.uint8(255*heatmap)

heatmap=cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)

exp_img = img2array.reshape(224,224,3)
z=heatmap*0.4+exp_img
plt.imshow(z/255)
plt.show()

