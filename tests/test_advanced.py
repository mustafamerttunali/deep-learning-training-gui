import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import IPython.display as display
from PIL import Image
import subprocess
import pathlib
np.random.seed(0)
tf_version = tf.__version__

if tf_version < "2.0.0":
    subprocess.call(['pip', 'install', 'tensorflow-gpu'])
else:
    print("Your TensorFlow version is up to date! {}".format(tf_version))