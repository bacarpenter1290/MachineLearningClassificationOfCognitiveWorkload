# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 21:26:03 2020

@author: Baylee
"""

url = 'http://127.0.0.1:5000/Test'

import requests

response = requests.get(url)
response.json()