import numpy as np
from flask import Flask, request, jsonify, render_template
from dltgui.dlgui import dl_gui
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


@app.route('/terminal',methods = ['POST', 'GET'])
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
      #gui = dl_gui(dataset=dataset, split_dataset = float(split_dataset), pre_trained_model = pre_trained_model, number_of_classes = int(number_of_classes), batch_size = int(batch_size), epoch = int(epoch) )
      #gui.load_dataset()
      #gui = gui.train()
      return render_template("terminal.html",result = result)


@app.route('/update')
def val():
    pass


if __name__ == "__main__":
    app.run(debug=True)
   
   