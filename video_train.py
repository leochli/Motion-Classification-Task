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

__all__ = ['Inception3', 'inception_v3']


lr = 0.05
momentum = 0.9
log_interval = 10
epochs = 200

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

# #Naive SVM
# clf = svm.SVC()
# clf.fit(X_train, y_train)
# print(clf.score(X_test, y_test))  

# #Store the parameter for enxt time
# joblib.dump(clf, 'fullfeature_2p.pkl') 


# If trained before, reuse the storing parameter for the classifier
# clf = joblib.load('fullfeature_2p.pkl') 


# print(clf.score(X_test, y_test))

x_train_tensor = torch.from_numpy(X_train[:, newaxis,:])
y_train_tensor = torch.from_numpy(y_train)
x_test_tensor = torch.from_numpy(X_test[:, newaxis,:])
y_test_tensor = torch.from_numpy(y_test)


train_dataset = data_utils.TensorDataset(x_train_tensor,y_train_tensor)
train_loader = data_utils.DataLoader(train_dataset, batch_size=100 * 4, shuffle=True)

test_dataset = data_utils.TensorDataset(x_test_tensor,y_test_tensor)
test_loader = data_utils.DataLoader(test_dataset, batch_size=50, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5)
	self.conv3 = nn.Conv1d(128, 256, kernel_size=3)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(13824, 50)
        self.fc2 = nn.Linear(50, 14)

    def forward(self, x):
        x = F.relu(F.max_pool1d(self.conv1(x), 2))
	x = F.relu(F.max_pool1d(self.conv2(x), 2))
        x = F.relu(F.max_pool1d(self.conv3_drop(self.conv3(x)), 2))
        # print(x.size())
        x = x.view(-1, 13824)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

class AlexNet(nn.Module):

    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(3328, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 14),
        )

    def forward(self, x):
        x = self.features(x)
        # print(x.size())
        x = x.view(x.size(0), 3328)
        x = self.classifier(x)
        return F.log_softmax(x)

model = Net()
model.cuda()

#optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
#scheduler = MultiStepLR(optimizer, milestones=[50,100,150], gamma=0.1)
optimizer = torch.optim.Adagrad(model.parameters(), lr=0.005, lr_decay=0, weight_decay=0)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # if args.cuda:
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        # if args.cuda:
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    
    test_loss /= len(test_loader.dataset)
    with open('custom_adagradv2_acc.txt','a') as f:
	f.write('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

for epoch in range(1, epochs + 1):
    train(epoch)
#    optimizer.step()
    test()





