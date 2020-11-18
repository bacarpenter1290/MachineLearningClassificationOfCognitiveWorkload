# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 16:47:15 2020

@author: Baylee
for ECE 429 and 439
at Purdue University Northwest
Electrical and Computer Engineering Department
"""
import wfdb as wf
import numpy as np
import graphviz
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
from sklearn import tree
from sklearn import preprocessing

signals6, fields6 = wf.io.rdsamp("drive06")
signals7, fields7 = wf.io.rdsamp("drive07")
signals8, fields8 = wf.io.rdsamp("drive08")
signals11, fields11 = wf.io.rdsamp("drive11")
signals10, fields10 = wf.io.rdsamp("drive10")
print(signals6[:,0])
print(fields6)
   
# plot signals 
plt.figure(1)
plt.subplot(211)
plt.plot(signals6[:,0])
plt.plot(signals6[:,1])
plt.plot(signals6[:,2])
plt.plot(signals6[:,3])
plt.plot(signals6[:,4])
plt.plot(signals6[:,6])
plt.legend('ECG', 'EMG', 'FGSR', 'HGSR', 'HR', 'RESP')
plt.title('Signals and markers from drive 6')

# plot markers
plt.figure(1)
plt.subplot(212)
plt.plot(signals6[:,0])

# set section length for train data
sectionLength = 50

# set starting points for each period
restStart6 = 4000
cityStart6 = 37000
HWStart6 = 45000

restStart8 = 4000
cityStart8 = 28000
HWStart8 = 45000

restStart10 = 4000
cityStart10 = 20000
HWStart10 = 32000

restStart11 = 4000
cityStart11 = 20000
HWStart11 = 31000

# create separate lists for signals and labels for each region
trainDataRest6 = signals6[restStart6:restStart6 + sectionLength][:]
trainLabelsRest6 = np.ones(len(trainDataRest6))

trainDataCity6 = signals6[cityStart6:cityStart6 + sectionLength][:]
trainLabelsCity6 = np.ones(len(trainDataCity6)) * 3

trainDataHW6 = signals6[HWStart6:HWStart6 + sectionLength][:]
trainLabelsHW6 = np.ones(len(trainDataHW6)) * 2

trainDataRest8 = signals8[restStart8:restStart8 + sectionLength][:]
trainLabelsRest8 = np.ones(len(trainDataRest8))

trainDataCity8 = signals8[cityStart8:cityStart8 + sectionLength][:]
trainLabelsCity8 = np.ones(len(trainDataCity8)) * 3

trainDataHW8 = signals8[HWStart8:HWStart8 + sectionLength][:]
trainLabelsHW8 = np.ones(len(trainDataHW8)) * 2

trainDataRest10 = signals10[restStart10:restStart10 + sectionLength][:]
trainLabelsRest10 = np.ones(len(trainDataRest10))

trainDataCity10 = signals10[cityStart10:cityStart10 + sectionLength][:]
trainLabelsCity10 = np.ones(len(trainDataCity10)) * 3

trainDataHW10 = signals10[HWStart10:HWStart10 + sectionLength][:]
trainLabelsHW10 = np.ones(len(trainDataHW10)) * 2

trainDataRest11 = signals11[restStart11:restStart11 + sectionLength][:]
trainLabelsRest11 = np.ones(len(trainDataRest11))

trainDataCity11 = signals11[cityStart11:cityStart11 + sectionLength][:]
trainLabelsCity11 = np.ones(len(trainDataCity11)) * 3

trainDataHW11 = signals11[HWStart11:HWStart11 + sectionLength][:]
trainLabelsHW11 = np.ones(len(trainDataHW11)) * 2


# combine regions into one training set
length = sectionLength * 3

trainData = [[] for i in range(4*length)]
trainLabels = []
for i in range(7):
    for j in range(length):
        if j < sectionLength:
            trainData[j].append(trainDataRest6[j][i])
        elif j >= sectionLength and j < 2*sectionLength:
            trainData[j].append(trainDataCity6[j-sectionLength][i])
        else:
            trainData[j].append(trainDataHW6[j-(2*sectionLength)][i])
     
for i in range(7):
    for j in range(length):
        if j < sectionLength:
            trainData[j+length].append(trainDataRest8[j][i])
        elif j >= sectionLength and j < 2*sectionLength:
            trainData[j+length].append(trainDataCity8[j-sectionLength][i])
        else:
            trainData[j+length].append(trainDataHW8[j-(2*sectionLength)][i])
   
for i in range(7):
    for j in range(length):
        if j < sectionLength:
            trainData[j+(2*length)].append(trainDataRest10[j][i])
        elif j >= sectionLength and j < 2*sectionLength:
            trainData[j+(2*length)].append(trainDataCity10[j-sectionLength][i])
        else:
            trainData[j+(2*length)].append(trainDataHW10[j-(2*sectionLength)][i])
   
for i in range(7):
    for j in range(length):
        if j < sectionLength:
            trainData[j+(3*length)].append(trainDataRest11[j][i])
        elif j >= sectionLength and j < 2*sectionLength:
            trainData[j+(3*length)].append(trainDataCity11[j-sectionLength][i])
        else:
            trainData[j+(3*length)].append(trainDataHW11[j-(2*sectionLength)][i])
      
for j in range(length):
    if j < sectionLength:
        trainLabels.append(trainLabelsRest6[j])
    elif j >= sectionLength and j < 2*sectionLength:
        trainLabels.append(trainLabelsCity6[j-sectionLength])
    else:
        trainLabels.append(trainLabelsHW6[j-(2*sectionLength)])

for j in range(length):
    if j < sectionLength:
        trainLabels.append(trainLabelsRest8[j])
    elif j >= sectionLength and j < 2*sectionLength:
        trainLabels.append(trainLabelsCity8[j-sectionLength])
    else:
        trainLabels.append(trainLabelsHW8[j-(2*sectionLength)])
  
for j in range(length):
    if j < sectionLength:
        trainLabels.append(trainLabelsRest10[j])
    elif j >= sectionLength and j < 2*sectionLength:
        trainLabels.append(trainLabelsCity10[j-sectionLength])
    else:
        trainLabels.append(trainLabelsHW10[j-(2*sectionLength)])

for j in range(length):
    if j < sectionLength:
        trainLabels.append(trainLabelsRest11[j])
    elif j >= sectionLength and j < 2*sectionLength:
        trainLabels.append(trainLabelsCity11[j-sectionLength])
    else:
        trainLabels.append(trainLabelsHW11[j-(2*sectionLength)])
   
# create testing set from drive 7     
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
  
# remove unused features (physiological data device doesn't collect these signals)
testData = np.delete(testData, 5, 1)
testData = np.delete(testData, 1, 1)
testData = np.delete(testData, 1, 1)
testData = np.delete(testData, 0, 1)

trainData = np.delete(trainData, 5, 1)
trainData = np.delete(trainData, 1, 1)
trainData = np.delete(trainData, 1, 1)
trainData = np.delete(trainData, 0, 1)

signals7_del = np.delete(signals7, 5, 1)
signals7_del = np.delete(signals7_del, 1, 1)
signals7_del = np.delete(signals7_del, 1, 1)
signals7_del = np.delete(signals7_del, 0, 1)

#print(testData)
#print(trainLabels)

trainLabels_np = np.array(trainLabels)

# normalize the data
# didn't work as well as scaling, didn't use in final version
trainData_norm = normalize(trainData,axis = 0)
plt.figure(3)
plt.plot(trainData_norm)
plt.title("Norm train data")
#plt.plot(trainLabels_np)

testData_norm = normalize(testData,axis = 0)
plt.figure(4)
plt.plot(testData_norm)
plt.title("Norm test data")
#plt.plot(testLabels)


# scale data
trainData_scale = preprocessing.scale(trainData)
testData_scale = preprocessing.scale(testData)

plt.figure(5)
plt.plot(trainData_scale)
plt.title("scale train data")

plt.figure(6)
plt.plot(testData_scale)
plt.title("scale test data")

plt.figure(7)

# split the training set into a training dataset and a validation set
train_in, val_in, train_out, val_out = model_selection.train_test_split(trainData_scale, 
                                                                        trainLabels, test_size=0.2)


# fit the classifiers
# decision tree was chosen. the rest are commented out to decrease run time
dtc = DecisionTreeClassifier()
tree.plot_tree(dtc.fit(train_in, train_out))

#svc = SVC(gamma='auto')
#svc.fit(train_in, train_out)
#
#mlp = MLPClassifier(max_iter=3*sectionLength)
#mlp.fit(train_in, train_out)
#
#km = KMeans()
#km.fit(train_in, train_out)


#dtc.fit(trainData, trainLabels_np)

#plot_decision_regions(trainData, trainLabels_np , clf=dtc)

#print(svc.score(testData_norm, testLabels))

#print(svc.predict(signals7_del[5000:6000]))
#print(svc.predict(signals7_del[20000:21000]))
#print(svc.predict(signals7_del[35000:36000]))

#print(mlp.score(testData_norm, testLabels))

#print(mlp.predict(signals7_del[5000:6000]))
#print(mlp.predict(signals7_del[20000:21000]))
#print(mlp.predict(signals7_del[35000:36000]))

#print(km.score(testData_norm, testLabels))
#print("Actual")
#print(val_out)
#print("predictions: ")

#print(dtc.predict(val_in))
#print(dtc.predict(signals7_del[5000:6000]))
#print(dtc.predict(signals7_del[20000:21000]))
#print(dtc.predict(signals7_del[35000:36000]))

# validate classifier using score
print("Accuracy: ")
print(dtc.score(val_in, val_out))


# test accuracy of classifier using predict
total1 = 0
total2 = 0
total3 = 0

testRange = 1000 

for i in range(testRange):
    total1 += dtc.predict(testData_scale[1000 + i].reshape(1,-1))
    total2 += dtc.predict(testData_scale[7000 + i].reshape(1,-1))
    total3 += dtc.predict(testData_scale[12000 + i].reshape(1,-1))
   
avg1 = total1 / testRange
avg2 = total2 / testRange
avg3 = total3 / testRange
print("Average class during rest ", avg1)
print("Average class during city ", avg2)
print("Average class during highway ", avg3)

# test accuracy of classifier using score
print(dtc.score(testData_scale, testLabels))

# plot the resulting decision tree
feature_names = ["Foot GSR", "HR", "RESP"]
target_names = ["Underworked", "Working Efficiently", "Overworked"]

dot_data = tree.export_graphviz(dtc, out_file=None,
                                feature_names=feature_names,  
                                class_names=target_names,
                                filled=True, rounded=True,  
                                special_characters=True) 
graph = graphviz.Source(dot_data) 

# save the classifier to a folder
from joblib import dump
dump(dtc, 'classifier.joblib')

# render the tree into a pdf file
#graph.render("Decision Tree - Test")