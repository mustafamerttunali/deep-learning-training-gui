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
    return render_template('index.html', title ="Version 1.0.2")

@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/training')
def training():
    return render_template('training.html')

@app.route('/terminal-object-detection',methods = ['POST'])
def terminal_object_detection():
    if request.method == 'POST':
      result = request.form
      dataset = result['dataset']
      type_of_label = result['type_of_label']
      split_dataset = result['split_dataset']
      project_name = result['project_name']
      pre_trained_model = result['Pre-trained Model']
      number_of_classes = result['noc']
      batch_size = result['batch_size']
      epoch = result['epoch']
      return render_template("terminal-object-detection.html",result = result)


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
      activation_function = result['activation_function']
      flip = result['flip'] 
      rotation = result['rotation'] 
      zoom = result['zoom']
      gui = dl_gui(dataset=dataset, 
                   project_name = str(project_name), 
                   split_dataset = float(split_dataset), 
                   pre_trained_model = pre_trained_model,
                   cpu_gpu = cpu_gpu,
                   number_of_classes = int(number_of_classes), 
                   batch_size = int(batch_size), 
                   epoch = int(epoch),
                   activation_function = activation_function)

      if(flip == "True" or rotation == "True" or zoom == "True"):
          samples = result['samples']
          gui.load_dataset(imgaugmentation = True, flip = flip, rotation = rotation, zoom = zoom, samples = int(samples))
      else:
          gui.load_dataset()
        
      thread_gui = threading.Thread(target= gui.train).start()
      return render_template("terminal.html",result = result, thread_gui = thread_gui)

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/result', methods = ['POST'])
def result():

    if request.method == 'POST':
        result = request.form
        dataset = result['dataset']
        model_dir= result['model']
        img = result['img'] 
        gui = dl_gui(project_name = "Test", dataset=dataset)
        predicted_class, max_pred = gui.predict(img, model_dir)
        return render_template('result.html', result = result, img = img, max_pred = max_pred, predicted_class = predicted_class,  mimetype="text/event-stream")

@app.route('/test', methods = ['POST'])
def test():
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
      activation_function = result['activation_function'] 
      flip = result['flip'] 
      rotation = result['rotation'] 
      zoom = result['zoom'] 
      print("Flip:   ", flip, rotation,zoom)
      return "Test Sayfası - terminale bak."



if __name__ == "__main__":
    app.run(debug=True)
   
   