# # # # # # # # # # # # # # # 
# Author: Mustafa Mert Tunalı
# ---------------------------
# ---------------------------
# Deep Learning Training GUI - Class Page
# ---------------------------
# ---------------------------
# Last Update: 31 December 2019

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
    def __init__(self, project_name, dataset, split_dataset = 0.20, pre_trained_model = 'MobileNetV2', cpu = 0, gpu = 1, number_of_classes = 5, batch_size = 16, epoch = 1):
         self.project_name = project_name
         self.data_dir = pathlib.Path(dataset)
         self.split_dataset = split_dataset
         self.pre_trained_model = pre_trained_model
         self.cpu = cpu
         self.gpu = gpu
         self.noc = number_of_classes
         self.batch_size = batch_size
         self.epoch = epoch
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
    

      
    def train(self):

        if self.pre_trained_model == "MobileNetV2":
            "Image should be (96, 96), (128, 128), (160, 160),(192, 192), or (224, 224)"
            mobilenet = tf.keras.applications.MobileNetV2(input_shape = (224,224,3),
                                                            include_top=False, 
                                                            weights='imagenet')
            mobilenet.trainable = False
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
            prediction_layer = tf.keras.layers.Dense(self.noc, activation = 'softmax')
            global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
            model = tf.keras.Sequential([
            VGG16,
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

        elif self.pre_trained_model == "VGG19":
            VGG19 = tf.keras.applications.VGG19(input_shape = (224,224,3),
                                                            include_top=False, 
                                                            weights='imagenet')
            VGG19.trainable = False
            prediction_layer = tf.keras.layers.Dense(self.noc, activation = 'softmax')
            global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
            model = tf.keras.Sequential([
            VGG19,
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


        elif self.pre_trained_model == "NASNetMobile":
            NASNetMobile = tf.keras.applications.NASNetMobile(input_shape = (224,224,3),
                                                            include_top=False, 
                                                            weights='imagenet')
            NASNetMobile.trainable = False
            prediction_layer = tf.keras.layers.Dense(self.noc, activation = 'softmax')
            global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
            model = tf.keras.Sequential([
            NASNetMobile,
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
                Dense(5, activation='softmax')
            ])

            model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

            history = model.fit_generator(
                self.train_data_gen,
                steps_per_epoch=self.STEPS_PER_EPOCH,
                validation_data = self.test_data_gen,
                validation_steps = self.VALID_STEPS_PER_EPOCH,
                epochs=self.epoch)
            
   
    

   

    
