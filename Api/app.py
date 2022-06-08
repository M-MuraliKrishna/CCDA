from flask import Flask, jsonify, request
from flask.helpers import send_from_directory
from flask.wrappers import Response
from flask_cors import CORS, cross_origin

import os
import json
import pickle
import cv2
import numpy as np
from scipy.stats import kurtosis, skew
from scipy import ndimage
import skimage.measure
import pandas as pd



app = Flask(__name__,static_folder='final/build',static_url_path='')

CORS(app)  


model=pickle.load(open('model.pkl','rb'))


@app.route('/predict', methods=['GET','POST'])
def predict():
        try:
                
                files = request.files
                
                file = files.get('file')
                print(files)
                response = file.read()
                frame = cv2.imdecode(np.fromstring(response, np.uint8), cv2.IMREAD_COLOR)
                # Response.headers.add('Access-Control-Allow-Origin', '*')
                # print(form_values)
                # path=str("C:/Users/adhyansh/Downloads/")+str(form_values['name'])
                # print(path)
                simg=frame
                print(simg,type(simg))
                sampleimg=[]
                simg = cv2.cvtColor(simg, cv2.COLOR_BGR2GRAY)
                simg = cv2.resize(simg, (600, 300), interpolation = cv2.INTER_AREA)
                simg = simg - simg.mean()    

                sampleimg.append((np.var(simg),skew(simg,axis=None),kurtosis(simg,axis=None),skimage.measure.shannon_entropy(simg)))

                v,s,k,e=sampleimg[0]
                new_banknote = pd.DataFrame({'variance':[v],'skewness':[s],'kurtosis':[k],'entropy':[e]})
                if model.predict(new_banknote)==1:
                    response = jsonify("Note is Real")
                    response.headers.add("Access-Control-Allow-Origin", "*")
                    return response
                    # return json.dumps("note is real")
                else:
                    response = jsonify("Note is Fake")
                    response.headers.add("Access-Control-Allow-Origin", "*")
                    return response
                    # return json.dumps("Note is fake")
        except:
                response = jsonify("Response Not Found")
                response.headers.add("Access-Control-Allow-Origin", "*")
                return response

@app.route('/')
def serve():
            return send_from_directory(app.static_folder,'index.html')

if __name__ == "__main__":
    app.run(debug=True)
