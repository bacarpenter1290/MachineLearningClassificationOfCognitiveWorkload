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

app = Flask(__name__)
api = Api(app)

model = load('C:\\Users\\Baylee\\Documents\\ML\\classifier.joblib')

parser = reqparse.RequestParser()
parser.add_argument('HR')
parser.add_argument('RESP')
parser.add_argument('GSR')

class PredictStress(Resource):
    def post(self):
        try:
            args = parser.parse_args()
            HRData = args['HR']
            RESPData = args['RESP']
            GSRData = args['GSR']
            
            inputArray = np.array([GSRData, HRData, RESPData])
            prediction = model.predict(inputArray.reshape(1,-1))
            
            if prediction == 1:
                pred_text = 'Understressed'
            elif prediction == 2:
                pred_text = 'Moderately Stressed'
            elif prediction == 3:
                pred_text = 'Overstressed'
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