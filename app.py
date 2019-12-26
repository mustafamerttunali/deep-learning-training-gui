import numpy as np
from multiprocessing import Process
import threading
from flask import Flask, request, jsonify, render_template
from dltgui.dlgui import dl_gui
import tensorflow as tf
# Set Flask
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/training')
def training():
    return render_template('training.html')

@app.route('/terminal',methods = ['POST'])
def terminal():

   if request.method == 'POST':
      '''Read the values from HTML file and set the values for training.''' 
      result = request.form
      dataset = result['dataset']
      split_dataset = result['split_dataset']
      project_name = result['project_name']
      pre_trained_model = result['Pre-trained Model']
      cpu_gpu = result['CPU/GPU']
      number_of_classes = result['noc']
      batch_size = result['batch_size']
      epoch = result['epoch']
      gui = dl_gui(dataset=dataset, split_dataset = float(split_dataset), pre_trained_model = pre_trained_model, number_of_classes = int(number_of_classes), batch_size = int(batch_size), epoch = int(epoch))
      gui.load_dataset()
      thread_gui = threading.Thread(target= gui.train).start()
      return render_template("terminal.html",result = result, thread_gui = thread_gui)

@app.route('/predict')
def predict():

    return render_template('predict.html')

@app.route('/result', methods = ['POST'])
def result():
    def prepare_image(img):
        img = tf.keras.preprocessing.image.load_img(img, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array_expanded_dims = np.expand_dims(img_array, axis=0)
        return tf.keras.applications.mobilenet_v2.preprocess_input(img_array_expanded_dims)

    def sigmoid(x):
        s = 1 / (1 + np.exp(-x))
        return s

    if request.method == 'POST':
        result = request.form
        #dataset = result['dataset']
        model_dir= result['model']
        img = result['img']
        img2array = prepare_image(img)
        model = tf.keras.models.load_model('models/'+model_dir)
        pred = model.predict(img2array)
        s_pred = sigmoid(pred)
        max_pred = np.max(s_pred)
        print("En buyuk:", max_pred)
        gui = dl_gui(dataset='datasets/flower_photos')

        classes = gui.CLASS_NAMES
        
        
            
     

        
        return render_template('result.html', result = result, max_pred = max_pred, mimetype="text/event-stream")



if __name__ == "__main__":
    app.run(debug=True)
   
   