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
      return render_template("terminal.html",result = result)




if __name__ == "__main__":
    app.run(debug=True)
   