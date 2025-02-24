# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 21:01:57 2020

@author: Baylee
"""

from flask import Flask
import numpy as np
import json
from flask_restful import reqparse, Api, Resource
from joblib import load
import collections

app = Flask(__name__)
api = Api(app)

model = load('C:\\Users\\Baylee\\Documents\\GitHub\\MachineLearningClassificationOfCognitiveWorkload\\classifier.joblib')

parser = reqparse.RequestParser()
parser.add_argument('HR')
parser.add_argument('RESP')
parser.add_argument('GSR')

class_hist = collections.deque(maxlen=15)

class PredictStress(Resource):
    def post(self):
        try:
            args = parser.parse_args()
            HRData = args['HR']
            RESPData = args['RESP']
            GSRData = args['GSR']
            
            inputArray = np.array([GSRData, HRData, RESPData])
            prediction = model.predict(inputArray.reshape(1,-1))
            
            class_hist.append(prediction[0])
            
            total = 0
            for i in class_hist:
                total += i
            
            avg_pred = total / len(class_hist)
            
            if round(avg_pred) == 1:
                pred_text = 'Underworked'
            elif round(avg_pred) == 2:
                pred_text = 'Working Efficiently'
            elif round(avg_pred) == 3:
                pred_text = 'Overworked'
            else:
                pred_text = 'Unknown'
            
            output = {'class': pred_text}
        except:
            print("An exception has occured")
            output = {'class': 'unknown'}
        return output
    
class TestGet(Resource):
    def get(self):
        args = parser.parse_args()
        HRData = args['HR']
        RESPData = args['RESP']
        GSRData = args['GSR']
        output = {'HR': HRData, 'RESP': RESPData, 'GSR': GSRData}
        
        return output
    
api.add_resource(PredictStress, '/PredictStress')
api.add_resource(TestGet, '/Test')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)