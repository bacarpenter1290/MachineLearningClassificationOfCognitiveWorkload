# MachineLearningClassificationOfCognitiveWorkload
Code written for my 2020 senior design project for the Electrical and Computer Engineering Department at Purdue University Northwest. 

•	Senior Design.py (found in Classifier Training folder)

This is Python code that is used to train and test the classifiers discussed in this paper. Everything except for the decision tree classifier has been commented out to improve run time. The program also saved the decision tree classifier in a file called ‘classifier.joblib’, which is used by the web service to send classifications in real-time.


•	.dat and .hea files (found in Classifier Training folder)

These files are from a previous study [5] and contain the pre-existing data used to train and test the classifiers. The .dat files contain the signal data and the .hea files contain the headers, describing each feature.


•	gui.pde (found in gui folder)

This file is the Processing code for the graphical user interface (GUI) application. It is a modified version of the GUI that came with the HealthyPi. All of the other files found in the gui folder are supporting files for the GUI application.


•	DTC-Rest.py
This file is Python code that functions as a web service. The service accepts real-time data from the GUI and responds with a classification of the data. The web service uses the classifier saved in ‘classifier.joblib’ to classify incoming data.


•	ReadSerial.py

This file is Python code that reads the serial port that the Arduino GSR sensor is connected to. The code then writes the reading to a file that is then read by the GUI to display and send to the web service.


•	classifier.joblib

A serialized copy of the classifier generated in Senior Design.py.


•	output2.txt

The file that contains the output of ReadSerial.py and is read by the GUI application.
