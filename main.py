from flask import Flask, request, url_for, redirect, render_template
import pandas as pd 
import numpy as np
from pycaret.regression import *

app = Flask(__name__)

model = load_model('deployment_20072020')
cols = ['AT', 'V', 'AP', 'RH']

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/predict', methods = ['POST'])
def predict():
    features = [i for i in request.form.values()]
    af = np.array(features)
    unseen_data = pd.DataFrame([af], columns = cols)
    prediction = predict_model(model, data = unseen_data)
    output = int(prediction.Label[0])
    return render_template('home.html', pred = 'Expected net hourly electrical energy output is {0}MW.'.format(output))

if __name__ == '__main__':
    app.run(debug = True)