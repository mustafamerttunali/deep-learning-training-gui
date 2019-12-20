import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

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

@app.route('/settings')
def settings():
    return render_template('settings.html')

@app.route('/hyper-parameters')
def hyper_parameters():
    return render_template('hyper-parameters.html')



@app.route('/terminal',methods = ['POST', 'GET'])
def terminal():
   if request.method == 'POST':
      result = request.form
      dataset = result['dataset']
      split_dataset = result['split_dataset']
      project_name = result['project_name']
      pre_trained_model = result['Pre-trained Model']
      cpu_gpu = result['CPU/GPU']
      number_of_classes = result['noc']
      batch_size = result['batch_size']
      epoch = result['epoch']
      print(split_dataset)
      return render_template("terminal.html",result = result)




if __name__ == "__main__":
    app.run(debug=True)
    print("hello")
   