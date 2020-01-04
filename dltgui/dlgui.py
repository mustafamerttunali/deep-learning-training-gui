# # # # # # # # # # # # # # # 
# Author: Mustafa Mert Tunalı
# ---------------------------
# ---------------------------
# Deep Learning Training GUI - Class Page
# ---------------------------
# ---------------------------
# Last Update: 3 January 2020

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import IPython.display as display
from PIL import Image
import subprocess
from multiprocessing import Process
import datetime
import pathlib
import os

np.random.seed(0)
tf_version = tf.__version__


if tf_version < "2.0.0":
    subprocess.call(['pip', 'install', 'tensorflow-gpu'])
else:
    print("Your TensorFlow version is up to date! {}".format(tf_version))


def startTensorboard(logdir):
    # Start tensorboard with system call
    os.system("tensorboard --logdir {}".format(logdir))
   

class dl_gui:
    "Version 1.0 This version, allows you to train image classification model easily"
    def __init__(self, project_name, dataset, split_dataset = 0.20, pre_trained_model = 'MobileNetV2', cpu_gpu='', number_of_classes = 5, batch_size = 16, epoch = 1, activation_function =''):
         self.project_name = project_name
         self.data_dir = pathlib.Path(dataset)
         self.split_dataset = split_dataset
         self.pre_trained_model = pre_trained_model
         self.cpu_gpu = cpu_gpu
         self.noc = number_of_classes
         self.batch_size = batch_size
         self.epoch = epoch
         self.activation_function = activation_function
         self.IMG_HEIGHT, self.IMG_WIDTH = 224, 224
         self.CLASS_NAMES = np.array([item.name for item in self.data_dir.glob('*') if item.name != "LICENSE.txt"])

    def show_batch(self,image_batch, label_batch):
        plt.figure(figsize=(10,10))
        for n in range(16):
            ax = plt.subplot(5,5,n+1)
            plt.imshow(image_batch[n])
            plt.title(self.CLASS_NAMES[label_batch[n]==1][0].title())
            plt.axis('off')
        plt.show()   
  
    def load_dataset(self):
       
        image_count = len(list(self.data_dir.glob('*/*.jpg')))
        image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=self.split_dataset)
        self.STEPS_PER_EPOCH = np.ceil(image_count/self.batch_size)
        self.train_data_gen = image_generator.flow_from_directory(directory=str(self.data_dir),
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            target_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
                                                            classes = list(self.CLASS_NAMES),
                                                            subset='training')

        self.test_data_gen = image_generator.flow_from_directory(directory=str(self.data_dir),
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            target_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
                                                            classes = list(self.CLASS_NAMES),
                                                            subset='validation')

        self.VALID_STEPS_PER_EPOCH = np.ceil(self.test_data_gen.samples/self.batch_size)
        
        # image_batch, label_batch = next(train_data_gen)
        # self.show_batch(image_batch, label_batch)

    def prepare_image(self, img):
        img = tf.keras.preprocessing.image.load_img(img, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array_expanded_dims = np.expand_dims(img_array, axis=0)
        return tf.keras.applications.mobilenet_v2.preprocess_input(img_array_expanded_dims)

    def sigmoid(self, x):
        s = 1 / (1 + np.exp(-x))
        return s

    def predict(self, img, model_dir):
        img2array = self.prepare_image('static/'+ img)
        model = tf.keras.models.load_model('models/'+ model_dir)
        pred = model.predict(img2array)
        y_classes = pred.argmax(axis=-1)
        s_pred = self.sigmoid(pred)* 100
        max_pred = int(np.round(np.max(s_pred))) 
        classes = self.CLASS_NAMES
        return "".join(map(str, classes[y_classes])), max_pred

      
    def train(self):
        with tf.device(self.cpu_gpu):
            if self.pre_trained_model == "MobileNetV2":
                "Image should be (96, 96), (128, 128), (160, 160),(192, 192), or (224, 224)"
                mobilenet = tf.keras.applications.MobileNetV2(input_shape = (224,224,3),
                                                                include_top=False, 
                                                                weights='imagenet')
                mobilenet.trainable = False
                if self.noc == 1:
                    prediction_layer = tf.keras.layers.Dense(self.noc, activation = self.activation_function)
                    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
                    model = tf.keras.Sequential([
                    mobilenet,
                    global_average_layer,
                    prediction_layer
                    ])
                    model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
                else:
                    prediction_layer = tf.keras.layers.Dense(self.noc, activation = 'softmax')
                    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
                    model = tf.keras.Sequential([
                    mobilenet,
                    global_average_layer,
                    prediction_layer
                    ])

                    model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

                tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=0,
                            write_graph=True, write_images=False)
                
                    
                Process(target=startTensorboard, args=("logs",)).start()
                target=model.fit_generator(
                    self.train_data_gen,
                    steps_per_epoch=self.STEPS_PER_EPOCH,
                    validation_data = self.test_data_gen,
                    validation_steps = self.VALID_STEPS_PER_EPOCH,
                    epochs=self.epoch,
                    callbacks=[tensorboard])
                
                model.save('models/{}.h5'.format(self.project_name))

            elif self.pre_trained_model == "InceptionV3":
                inceptionv3 = tf.keras.applications.InceptionV3(input_shape = (224,224,3),
                                                                include_top=False, 
                                                                weights='imagenet')
                inceptionv3.trainable = False
                if self.noc == 1:
                    prediction_layer = tf.keras.layers.Dense(self.noc, activation = self.activation_function)
                    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
                    model = tf.keras.Sequential([
                    inceptionv3,
                    global_average_layer,
                    prediction_layer
                    ])
                    model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
                else:
                    prediction_layer = tf.keras.layers.Dense(self.noc, activation = 'softmax')
                    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
                    model = tf.keras.Sequential([
                    inceptionv3,
                    global_average_layer,
                    prediction_layer
                    ])

                    model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
                tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=0,
                            write_graph=True, write_images=False)
                Process(target=startTensorboard, args=("logs",)).start()
                target=model.fit_generator(
                    self.train_data_gen,
                    steps_per_epoch=self.STEPS_PER_EPOCH,
                    validation_data = self.test_data_gen,
                    validation_steps = self.VALID_STEPS_PER_EPOCH,
                    epochs=self.epoch,
                    callbacks=[tensorboard])
                
                model.save('models/{}.h5'.format(self.project_name))


            elif self.pre_trained_model == "VGG16":
                VGG16 = tf.keras.applications.VGG16(input_shape = (224,224,3),
                                                                include_top=False, 
                                                                weights='imagenet')
                VGG16.trainable = False
                if self.noc == 1:
                    prediction_layer = tf.keras.layers.Dense(self.noc, activation = self.activation_function)
                    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
                    model = tf.keras.Sequential([
                    VGG16,
                    global_average_layer,
                    prediction_layer
                    ])
                    model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
                else:
                    prediction_layer = tf.keras.layers.Dense(self.noc, activation = 'softmax')
                    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
                    model = tf.keras.Sequential([
                    VGG16,
                    global_average_layer,
                    prediction_layer
                    ])
                tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=0,
                            write_graph=True, write_images=False)
                Process(target=startTensorboard, args=("logs",)).start()
                target=model.fit_generator(
                    self.train_data_gen,
                    steps_per_epoch=self.STEPS_PER_EPOCH,
                    validation_data = self.test_data_gen,
                    validation_steps = self.VALID_STEPS_PER_EPOCH,
                    epochs=self.epoch,
                    callbacks=[tensorboard])
                
                model.save('models/{}.h5'.format(self.project_name))    

            elif self.pre_trained_model == "VGG19":
                VGG19 = tf.keras.applications.VGG19(input_shape = (224,224,3),
                                                                include_top=False, 
                                                                weights='imagenet')
                VGG19.trainable = False
                if self.noc == 1:
                    prediction_layer = tf.keras.layers.Dense(self.noc, self.activation_function)
                    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
                    model = tf.keras.Sequential([
                    VGG19,
                    global_average_layer,
                    prediction_layer
                    ])
                    model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
                else:
                    prediction_layer = tf.keras.layers.Dense(self.noc, activation = 'softmax')
                    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
                    model = tf.keras.Sequential([
                    VGG19,
                    global_average_layer,
                    prediction_layer
                    ])
                tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=0,
                            write_graph=True, write_images=False)
                Process(target=startTensorboard, args=("logs",)).start()
                target=model.fit_generator(
                    self.train_data_gen,
                    steps_per_epoch=self.STEPS_PER_EPOCH,
                    validation_data = self.test_data_gen,
                    validation_steps = self.VALID_STEPS_PER_EPOCH,
                    epochs=self.epoch,
                    callbacks=[tensorboard])
                
                model.save('models/{}.h5'.format(self.project_name)) 


            elif self.pre_trained_model == "NASNetMobile":
                NASNetMobile = tf.keras.applications.NASNetMobile(input_shape = (224,224,3),
                                                                include_top=False, 
                                                                weights='imagenet')
                NASNetMobile.trainable = False
                if self.noc == 1:
                    prediction_layer = tf.keras.layers.Dense(self.noc, self.activation_function)
                    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
                    model = tf.keras.Sequential([
                    NASNetMobile,
                    global_average_layer,
                    prediction_layer
                    ])
                    model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
                else:
                    prediction_layer = tf.keras.layers.Dense(self.noc, activation = 'softmax')
                    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
                    model = tf.keras.Sequential([
                    NASNetMobile,
                    global_average_layer,
                    prediction_layer
                    ])
                tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=0,
                            write_graph=True, write_images=False)
                Process(target=startTensorboard, args=("logs",)).start()
                target=model.fit_generator(
                    self.train_data_gen,
                    steps_per_epoch=self.STEPS_PER_EPOCH,
                    validation_data = self.test_data_gen,
                    validation_steps = self.VALID_STEPS_PER_EPOCH,
                    epochs=self.epoch,
                    callbacks=[tensorboard])
                
                model.save('models/{}.h5'.format(self.project_name)) 

            elif self.pre_trained_model == "SimpleCnnModel":
                if self.noc == 1:
                    model = Sequential([
                        Conv2D(16, 3, padding='same', activation='relu', input_shape=(self.IMG_HEIGHT, self.IMG_WIDTH ,3)),
                        MaxPooling2D(),
                        Conv2D(32, 3, padding='same', activation='relu'),
                        MaxPooling2D(),
                        Conv2D(64, 3, padding='same', activation='relu'),
                        MaxPooling2D(),
                        Flatten(),
                        Dense(512, activation='relu'),
                        Dense(self.noc, activation=self.activation_function)
                    ])
                    model.compile(optimizer='adam',
                        loss='binary_crossentropy',
                        metrics=['accuracy'])
                else:
                    model = Sequential([
                        Conv2D(16, 3, padding='same', activation='relu', input_shape=(self.IMG_HEIGHT, self.IMG_WIDTH ,3)),
                        MaxPooling2D(),
                        Conv2D(32, 3, padding='same', activation='relu'),
                        MaxPooling2D(),
                        Conv2D(64, 3, padding='same', activation='relu'),
                        MaxPooling2D(),
                        Flatten(),
                        Dense(512, activation='relu'),
                        Dense(self.noc, activation = 'softmax')
                    ])
                    model.compile(optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])

                tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=0,
                            write_graph=True, write_images=False)
                Process(target=startTensorboard, args=("logs",)).start()
                history = model.fit_generator(
                    self.train_data_gen,
                    steps_per_epoch=self.STEPS_PER_EPOCH,
                    validation_data = self.test_data_gen,
                    validation_steps = self.VALID_STEPS_PER_EPOCH,
                    epochs=self.epoch,
                    callbacks=[tensorboard])
                model.save('models/{}.h5'.format(self.project_name)) 
   
    

   

    
