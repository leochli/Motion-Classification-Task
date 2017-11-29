import csv
from sklearn import svm
from sklearn.externals import joblib
import pickle
import numpy as np
from numpy import zeros, newaxis

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torchvision
import torch.utils.data as data_utils
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.model_zoo as model_zoo


with open('output.csv','rb') as csvfile:
	spamreader = csv.reader(csvfile, delimiter=',')

	activities_list = []
	dictionary = {}
	test_dictionary = {}

	# To read the first person data and store the features
	for row in spamreader:

		if row[0] not in activities_list:
			activities_list.append(row[0])
			dictionary[row[0]] = []
		#1~459 features are skeletal features
		#dictionary[row[0]].append(np.append(np.asarray(row[1:652]),np.asarray(row[684:876])))
		dictionary[row[0]].append(np.asarray(row[1:460]))





with open('output_p2.csv','rb') as csvfile:
	spamreader2 = csv.reader(csvfile, delimiter=',')
	for row in spamreader2:

		if row[0] not in activities_list:
			activities_list.append(row[0])
			dictionary[row[0]] = []
		#dictionary[row[0]].append(np.append(np.asarray(row[1:652]),np.asarray(row[684:876])))
		dictionary[row[0]].append(np.asarray(row[1:460]))


with open('output3.csv','rb') as csvfile:
	spamreader3 = csv.reader(csvfile, delimiter=',')
	for row in spamreader3:
		if row[0] not in activities_list:
			activities_list.append(row[0])
			dictionary[row[0]] = []
		#dictionary[row[0]].append(np.append(np.asarray(row[1:652]),np.asarray(row[684:876])))
		dictionary[row[0]].append(np.asarray(row[1:460]))

with open('output4.csv','rb') as csvfile:
	spamreader4 = csv.reader(csvfile, delimiter=',')
	for row in spamreader4:
		if row[0] not in activities_list:
			activities_list.append(row[0])
			test_dictionary[row[0]] = []
		#test_dictionary[row[0]].append(np.append(np.asarray(row[1:652]),np.asarray(row[684:876])))

		test_dictionary[row[0]].append(np.asarray(row[1:460]))


label_dict = {}
act_dict = {}
x = []
y = []

with open("activityLabel2.txt", "r") as filestream:
	i = 0
	for line in filestream:
		currentline = line.split(",")
		# print('L' + currentline[0])
		act_dict['L' + currentline[0]] = currentline[1]
		if currentline[1] not in label_dict:
			label_dict[currentline[1]] = i
			i = i +1
with open("activityLabel.txt", "r") as filestream:
	for line in filestream:
		currentline = line.split(",")
		act_dict['L' + currentline[0]] = currentline[1]
with open("activityLabel3.txt", "r") as filestream:
	for line in filestream:
		currentline = line.split(",")
		act_dict['L' + currentline[0]] = currentline[1]
with open("activityLabel4.txt", "r") as filestream:
	for line in filestream:
		currentline = line.split(",")
		act_dict['L' + currentline[0]] = currentline[1]

print(label_dict)

for key in dictionary:
	# print(act_dict[key], key)
	# print(len(dictionary[key]))
	for feature in dictionary[key]:
		x.append(feature)
		y.append(label_dict[act_dict[key]])

x_test = []
y_test = []
i = 0


for key in test_dictionary:
	i = i + 1
	# print(act_dict[key], key)
	# print(len(test_dictionary[key]))
	for feature in test_dictionary[key]:
		#Unseen person
		if (True):
			x_test.append(feature)
			y_test.append(label_dict[act_dict[key]])
		else:
			x.append(feature)
			y.append(label_dict[act_dict[key]])


#X is samples of all the features and Y is their corresponding activities

X = np.asarray(x, dtype=np.float32)
Y = np.asarray(y, dtype=np.int32)

print(type(X),type(Y),X.shape,Y.shape)

#Split train and test half by half
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

X_train = X
y_train = Y
X_test = np.asarray(x_test,dtype=np.float32)
y_test = np.asarray(y_test, dtype=np.int32)

print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)

 #Naive SVM
clf = svm.SVC()
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))  

#Store the parameter for enxt time
joblib.dump(clf, 'fullfeature_2p.pkl') 


# If trained before, reuse the storing parameter for the classifier
# clf = joblib.load('fullfeature_2p.pkl') 


print(clf.score(X_test, y_test))
