# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 21:16:51 2020

@author: Hindrobo
"""


#!pip install flasgger

import pandas as pd 
import pickle
from flask import Flask, request
import flasgger
from flasgger import Swagger

app = Flask(__name__)
Swagger(app)
pickle_data = open("model_trained.pkl","rb")
trained_data = pickle.load(pickle_data)


@app.route('/')
def welcome():
    return "hello I'm kaggle"

@app.route('/prediction',methods=["Get"])
def predict_by_single_data_GET_method():
    
    """Let's Authenticate the Banks Note .
    ---
    parameters:  
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
    variance=request.args.get("variance")
    skewness=request.args.get("skewness")
    curtosis=request.args.get("curtosis")
    entropy=request.args.get("entropy")
    prediction=trained_data.predict([[variance,skewness,curtosis,entropy]])
    print(prediction)
    return "This is your prediction = "+str(prediction)
@app.route('/prediction_with_file',methods=["POST"])
def predict_by_file_POST_method():
    """Let's Authenticate the Banks Note .
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      
    responses:
        200:
            description: output values
        
    """
    df_test=pd.read_csv(request.files.get("file"))
    print(df_test.head())
    prediction=trained_data.predict(df_test)
    
    return str(list(prediction))
if __name__=='__main__':
    app.run()
# you need to paste this in your browser
#http://127.0.0.1:5000/apidocs/
