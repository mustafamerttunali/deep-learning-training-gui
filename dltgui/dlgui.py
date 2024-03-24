# # # # # # # # # # # # # # # 
# Author: Mustafa Mert Tunalı
# ---------------------------
# ---------------------------
# Deep Learning Training GUI - Class Page
# ---------------------------
# ---------------------------
# # # # # # # # # # # # # # # 


# Libraries
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np
import IPython.display as display
import cv2
from PIL import Image
import subprocess
from multiprocessing import Process
import datetime
import pathlib
import os
import Augmentor
import time

# makes the random numbers predictable.
np.random.seed(0)

# checking Tensorflow version
tf_version = tf.__version__

# TF Version lower than 2.0.0 could be a problem.
if tf_version < "2.1.0":
    subprocess.call(['pip', 'install', 'tensorflow-gpu'])
else:
    print("Your TensorFlow version is up to date! {}".format(tf_version))

# Tensorboard Function
def startTensorboard(logdir):
    # Start tensorboard with system call
    os.system("tensorboard --logdir {}".format(logdir))
   

class dl_gui:
    
    "Version 1.0 This version, allows you to train image classification model easily"
    def __init__(self, project_name, dataset, split_dataset = 0.20, pre_trained_model = 'MobileNetV2', cpu_gpu='', number_of_classes = 5, batch_size = 16, epoch = 1, activation_function ='', fine_tune_epochs = 10):
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
         self.fine_tune_epochs = fine_tune_epochs
         

         
    def show_batch(self,image_batch, label_batch):
        plt.figure(figsize=(10,10))
        for n in range(16):
            ax = plt.subplot(5,5,n+1)
            plt.imshow(image_batch[n])
            plt.title(self.CLASS_NAMES[label_batch[n]==1][0].title())
            plt.axis('off')
        plt.show()   
  
    def load_dataset(self, imgaugmentation = False, flip = False, rotation = False, zoom = False, samples = 100):
        if imgaugmentation == True:
            p = Augmentor.Pipeline(str(self.data_dir), output_directory="")
            if flip == "True":
                print("Flipping...")
                p.flip_left_right(probability = 0.5)
                
            if rotation == "True":
                print("Rotating...")
                p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)

            if  zoom == "True":
                print("Zooming...")
                p.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)  
            print("Total of Samples:  ", samples)
            p.sample(samples) 
            p.process()                  
        
        image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=self.split_dataset)
        self.train_data_gen = image_generator.flow_from_directory(directory=self.data_dir,
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            target_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
                                                            classes = list(self.CLASS_NAMES),
                                                            subset='training')
        self.STEPS_PER_EPOCH = np.ceil(self.train_data_gen.samples/self.batch_size)

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

    def prepare_image_heatmap(self, img):
            img = tf.keras.preprocessing.image.load_img(img, target_size=(224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            
            return img_array


    def show_heatmap(self, img, model, layer_name = ''):

        grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(np.array([img]))
            class_number = len(predictions[0])
            loss = predictions[:,class_number-1]

        output = conv_outputs[0]
        grads = tape.gradient(loss, conv_outputs)[0]

        weights = tf.reduce_mean(grads, axis=(0, 1))

        cam = np.ones(output.shape[0:2], dtype=np.float32)

        for index, w in enumerate(weights):
            cam += w * output[:, :, index]

        cam = cv2.resize(cam.numpy(), (224, 224))
        cam = np.maximum(cam, 0)
        heatmap = (cam - cam.min()) / (cam.max() - cam.min())

        cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)

        output_image = cv2.addWeighted(cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2BGR), 0.5, cam, 0.65, 0)

        filename = 'output_{}.png'.format(time.time())
        plt.imsave('static/'+ filename, output_image)
        return filename

    def sigmoid(self, x):
        s = 1 / (1 + np.exp(-x))
        return s

    
    def predict(self, img, model_dir):
        img2array = self.prepare_image('static/'+ img)
        img2array_heatmap = self.prepare_image_heatmap('static/'+ img)
        model = tf.keras.models.load_model('models/'+ model_dir)
        layer_name = 'Conv_1_bn' # For only MobileNetV2
        try:
            heat_map = self.show_heatmap(img2array_heatmap, model, layer_name)
            show_heatmap = True
        except:
            show_heatmap = False
            heat_map = None
            pass
        pred = model.predict(img2array)
        y_classes = pred.argmax(axis=-1)
        s_pred = self.sigmoid(pred)* 100
        max_pred = int(np.round(np.max(s_pred))) 
        classes = self.CLASS_NAMES
        return "".join(map(str, classes[y_classes])), max_pred, show_heatmap, heat_map

      
    def train(self, fine_tuning = "False"):
        "Image should be (96, 96), (128, 128), (160, 160),(192, 192), or (224, 224)"
        print("Detected GPU/CPU devices: ", tf.config.list_physical_devices())
        with tf.device(self.cpu_gpu):
            if self.pre_trained_model == "MobileNetV2":     
                mobilenet = tf.keras.applications.MobileNetV2(input_shape = (224,224,3),
                                                                include_top=False, 
                                                                weights='imagenet')
                mobilenet.trainable = False
                if self.noc == 2:
                    average_pooling = tf.keras.layers.GlobalAveragePooling2D()(mobilenet.output)
                    prediction_layer = tf.keras.layers.Dense(self.noc, activation = self.activation_function)(average_pooling)
                    model = tf.keras.Model(mobilenet.inputs, prediction_layer)
                    model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
                else:
                    average_pooling = tf.keras.layers.GlobalAveragePooling2D()(mobilenet.output)
                    prediction_layer = tf.keras.layers.Dense(self.noc, activation = 'softmax')(average_pooling)
                    model = tf.keras.Model(mobilenet.inputs, prediction_layer)
                    model.summary()
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
                
                if fine_tuning == "True":
                    mobilenet.trainable = True

                    # Fine-tune from this layer onwards
                    fine_tune_at = 100

                    # Freeze all the layers before the `fine_tune_at` layer
                    for layer in mobilenet.layers[:fine_tune_at]:
                        layer.trainable =  False

                    
                    total_epochs =   self.epoch + self.fine_tune_epochs
                    if self.noc == 2:
                        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001/10),
                        loss='binary_crossentropy',
                        metrics=['accuracy'])
                    else:
                        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001/10),
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
                    print("Fine-Tuning starting....")
                    history_fine = model.fit(self.train_data_gen,
                         epochs=total_epochs,
                         initial_epoch =  target.epoch[-1],
                         validation_data=self.test_data_gen)
                    model.save('models/{}_fine_tuned.h5'.format(self.project_name))

                else:
                    model.save('models/{}.h5'.format(self.project_name))
                        

            elif self.pre_trained_model == "InceptionV3":
                inceptionv3 = tf.keras.applications.InceptionV3(input_shape = (224,224,3),
                                                                include_top=False, 
                                                                weights='imagenet')
                inceptionv3.trainable = False
                if self.noc == 2:
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
                if self.noc == 2:
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
                if self.noc == 2:
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
                if self.noc == 2:
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

            elif self.pre_trained_model == "SimpleCnnModel":
                if self.noc == 2:
                    model = Sequential([
                        Conv2D(16, 3, padding='same', activation='relu', input_shape=(224,224,3)),
                        MaxPooling2D(),
                        Conv2D(32, 3, padding='same', activation='relu'),
                        MaxPooling2D(),
                        Conv2D(64, 3, padding='same', activation='relu'),
                        MaxPooling2D(),
                        Flatten(),
                        Dense(64, activation='relu'),
                        Dense(self.noc, activation = self.activation_function)
                    ])
                    model.compile(optimizer='adam',
                        loss='binary_crossentropy',
                        metrics=['accuracy'])
              
                else:
                    model = Sequential([
                        Conv2D(16, 3, padding='same', activation='relu', input_shape=(224,224,3)),
                        MaxPooling2D(),
                        Conv2D(32, 3, padding='same', activation='relu'),
                        MaxPooling2D(),
                        Conv2D(64, 3, padding='same', activation='relu'),
                        MaxPooling2D(),
                        Flatten(),
                        Dropout(0.5),
                        Dense(64, activation='relu'),
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
   
    

   

    
