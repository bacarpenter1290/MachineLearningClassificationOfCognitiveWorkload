# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 12:03:33 2020

@author: Baylee
"""

import wfdb as wf
import matplotlib.pyplot as plt
import numpy as np
import sklearn.model_selection as model_selection

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score

signals6, fields6 = wf.io.rdsamp("drive06")

print(fields6)
print(signals6)

ecg6_list = []
emg6_list = []
fgsr6_list = []
hgsr6_list = []
hr6_list = []
marker6_list = []
resp6_list = []

for i in range(len(signals6)):
    ecg6_list.append(signals6[i][0])
    emg6_list.append(signals6[i][1])
    fgsr6_list.append(signals6[i][2])
    hgsr6_list.append(signals6[i][3])
    hr6_list.append(signals6[i][4])
    marker6_list.append(signals6[i][5])
    resp6_list.append(signals6[i][6])
    
plt.figure(1)
plt.subplot(211)
plt.plot(ecg6_list)
plt.plot(emg6_list)
plt.plot(fgsr6_list)
plt.plot(hgsr6_list)
plt.plot(hr6_list)
plt.plot(resp6_list)
plt.legend(['ecg','emg','fgsr','hgsr','hr','resp'])

plt.figure(1)
plt.subplot(212)
plt.plot(marker6_list)

sectionLength = 100

restStart6 = 4000
cityStart6 = 37000
HWStart6 = 45000

trainDataRest6 = signals6[restStart6:restStart6 + sectionLength][:]
trainLabelsRest6 = np.ones(len(trainDataRest6))

trainDataCity6 = signals6[cityStart6:cityStart6 + sectionLength][:]
trainLabelsCity6 = np.ones(len(trainDataCity6)) * 3

trainDataHW6 = signals6[HWStart6:HWStart6 + sectionLength][:]
trainLabelsHW6 = np.ones(len(trainDataHW6)) * 2

length = sectionLength * 3

trainData = [[] for i in range(length)]
trainLabels = []
for i in range(7):
    for j in range(length):
        if j < sectionLength:
            trainData[j].append(trainDataRest6[j][i])
        elif j >= sectionLength and j < 2*sectionLength:
            trainData[j].append(trainDataCity6[j-sectionLength][i])
        else:
            trainData[j].append(trainDataHW6[j-(2*sectionLength)][i])

for j in range(length):
    if j < sectionLength:
        trainLabels.append(trainLabelsRest6[j])
    elif j >= sectionLength and j < 2*sectionLength:
        trainLabels.append(trainLabelsCity6[j-sectionLength])
    else:
        trainLabels.append(trainLabelsHW6[j-(2*sectionLength)])
        
trainData = np.delete(trainData, 5, 1)

train_in, val_in, train_out, val_out = model_selection.train_test_split(trainData, trainLabels, test_size = 0.2)

class Model:
    def __init__(self, genericModel, name):
        self.genericModel = genericModel
        self.name = name

#models = []
#svc = SVC(gamma='auto')
#svc.fit(train_in, train_out)
#models.append(Model(svc, 'svc'))
dtc = DecisionTreeClassifier()
dtc.fit(train_in, train_out)
#models.append(Model(dtc, 'dtc'))
#mlp = MLPClassifier(max_iter=1000)
#mlp.fit(train_in, train_out)
#models.append(Model(mlp, 'mlp'))

#import time
#
#for model in models:
#    accuracyList = []
#    stdList = []
#    timeList = []
#    for i in range(100):
#        start_time = time.time()
#        cv_score = cross_val_score(model.genericModel, val_in, val_out, cv=5, scoring='accuracy')
#        #print(model.name, cv_score.mean(), cv_score.std())
#        accuracyList.append(cv_score.mean())
#        stdList.append(cv_score.std())
#        run_time = time.time() - start_time
#        timeList.append(run_time)
#    accuracy = sum(accuracyList) / len(accuracyList)
#    std = sum(stdList) / len(stdList)
#    avgTime = sum(timeList) / len(timeList)
#    print(model.name, accuracy, std, avgTime)


signals7, fields7 = wf.io.rdsamp("drive07")

testDataRest = signals7[4000:10000][:]
testLabelsRest = np.ones(len(testDataRest))

testDataCity = signals7[20000:25000][:]
testLabelsCity = np.ones(len(testDataCity)) * 3

testDataHW = signals7[33000:36000][:]
testLabelsHW = np.ones(len(testDataHW)) * 2

length = 14000

testData = [[] for i in range(length)]
testLabels = []
for i in range(7):
    for j in range(length):
        if j < 6000:
            testData[j].append(testDataRest[j][i])
        elif j >= 6000 and j < 11000:
            testData[j].append(testDataCity[j-6000][i])
        else:
            testData[j].append(testDataHW[j-11000][i])
            
for j in range(length):
    if j < 6000:
        testLabels.append(testLabelsRest[j])
    elif j >= 6000 and j < 11000:
        testLabels.append(testLabelsCity[j-6000])
    else:
        testLabels.append(testLabelsHW[j-11000])
        
testData = np.delete(testData, 5, 1)

print('DTC test score:', dtc.score(testData, testLabels))


from sklearn import preprocessing
trainData_scale = preprocessing.scale(trainData)
testData_scale = preprocessing.scale(testData)

train_in, val_in, train_out, val_out = model_selection.train_test_split(trainData_scale, trainLabels, test_size = 0.2)

dtc_scale = DecisionTreeClassifier()
dtc_scale.fit(train_in, train_out)

print('DTC test score after scaling:', dtc_scale.score(testData_scale, testLabels))