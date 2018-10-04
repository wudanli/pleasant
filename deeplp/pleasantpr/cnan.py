
# coding: utf-8

# In[ ]:


import pandas as pd
import torch
import copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
 
batch_size = 14
# Data_Xlsx 
def Data_Reading(Normalization = True):
    # Read the xlsx
    train_x = pd.read_excel( "trainingset.xlsx", 'Input',header = None)
    test_x = pd.read_excel("oilsset.xlsx",'oils',header = None)
    test_z = pd.read_excel("newodorset.xlsx",'new',header = None)
    train_y = pd.read_excel("trainingy1.xlsx",'Output',header = None)
    # Normalization
    train_x_Normed = copy.deepcopy(train_x).apply(lambda x : (x - np.min(x)) / (np.max(x) - np.min(x)))
    test_x_Normed = copy.deepcopy(test_x).apply(lambda x : (x - np.min(x)) / (np.max(x) - np.min(x)))
    test_z_Normed = copy.deepcopy(test_z).apply(lambda x : (x - np.min(x)) / (np.max(x) - np.min(x)))
    train_y_Normed = copy.deepcopy(train_y).apply(lambda x : (x - np.min(x)) / (np.max(x) - np.min(x)))
    # xlsx to tensor
    if Normalization:
        train_x = torch.from_numpy(train_x_Normed.values).type(torch.FloatTensor)
        test_x = torch.from_numpy(test_x_Normed.values).type(torch.FloatTensor)
        test_z = torch.from_numpy(test_z_Normed.values).type(torch.FloatTensor)
        train_y = torch.from_numpy(train_y_Normed.values).type(torch.LongTensor)
    else:
        train_x = torch.from_numpy(train_x.values).type(torch.FloatTensor)
        test_x = torch.from_numpy(test_x.values).type(torch.FloatTensor)
        test_z = torch.from_numpy(test_z.values).type(torch.FloatTensor)
        train_y = torch.from_numpy(train_y.values).type(torch.LongTensor)
    #reshape
    train_x= train_x.view(238,1,16,1496)
    test_x= test_x.view(108,1,16,1496)
    test_z= test_z.view(95,1,16,1496)
    train_y=  train_y.squeeze()
    return train_x, test_x, test_z, train_y


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #input shape (1,16,1496)
        self.conv1 = nn.Conv2d(1,6,(3,5)) #con2d出来结果大小不变output shape (6,16,1496)
        self.conv2 = nn.Conv2d(6,10,(3,5)) 
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2) 
        self.mp2 =  nn.MaxPool2d(kernel_size=3, stride=2)
        self.fc = nn.Linear(10*2*370,10)   
             
    def forward(self, x):
        x = F.relu(self.mp1(self.conv1(x)))#（238,6,9，749）
        x = F.relu(self.mp2(self.conv2(x)))#（238,16,5，375）
        x = x.view(x.size(0), -1) # flatten the tensor
        x = self.fc(x)
        return x


cnn = Net()
print(cnn)

optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)
loss_func = nn.CrossEntropyLoss()

#training
def train(train_x,train_y):
    for epoch in range(5):
        t_x = Variable(train_x)
        t_y = Variable(train_y)
        #forward
        out = cnn(t_x)
        loss = loss_func(out, t_y)
        optimizer.zero_grad()
        #backward
        loss.backward()
        optimizer.step()
        print('Epoch[{}/{}], loss: {:.12f},'.format(epoch + 1,5, loss.item()))
            
#predicting 
def predict(test_x,test_z):
    for epoch in range(5):
        te_x = Variable(test_x)
        te_z = Variable(test_z)
        out1 = cnn(te_x)
        out2 = cnn(te_z)
        print('epoch',epoch+1,out1)
    

train_x, test_x, test_z, train_y = Data_Reading(Normalization = True)
train(train_x,train_y)
predict(test_x,test_z)

