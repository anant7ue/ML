#!/usr/bin/python3

import pandas as pd
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as nnFunc

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error

nArg = len(sys.argv)
if(nArg < 4):
    print("Too few arguments: ", nArg)
    print("Usage: python3 script.py [NumEpochs] [LearnRate] [NumStages] ")
    exit(0)

numParam = 6
debug = False

# hyperparameter value is used to control the learning process.

numEpoch = int(sys.argv[1])
lRate = float(sys.argv[2])
nStage = int(sys.argv[3])
print("numEpochs=" , numEpoch, "learning rate=", lRate, "num stages=", nStage)

# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html

td = pd.read_csv('trainData2.csv')
yRespTime = td['RespTime'].copy()
yTputIops = td['ThputIOps'].copy()
yTputMbps = td['ThputMbps'].copy()
X = td.drop(['RespTime', 'ThputIOps', 'ThputMbps'], axis=1).copy()
if(debug):
    print("cols=", td.columns)
    print("")

x_train, x_test, yrt_train, yrt_test = train_test_split(X, yRespTime, train_size=0.75, random_state=1)

scaler = StandardScaler()

if(debug):
    print("xtrain=", x_train)
scaledX = scaler.fit_transform(x_train)
if(debug):
    print("scaledX= ", scaledX[0])
    print("")
scaledXTest = scaler.fit_transform(x_test)
if(debug):
    print("scaledX test= ", scaledXTest)

x_train = torch.tensor(scaledX).type(torch.float32)
y_train = torch.tensor(np.array(yrt_train)).type(torch.float32)

x_test = torch.tensor(scaledXTest).type(torch.float32)
y_test = torch.tensor(np.array(yrt_test)).type(torch.float32)

# https://pytorch.org/docs/stable/generated/torch.nn.Module.html

class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()

        # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html

        self.layer1 = nn.Linear(numParam, nStage)
        self.layer2 = nn.Linear(nStage, nStage)
        self.out = nn.Linear(nStage, 1)

    def forward(self, n):
        n = nnFunc.relu(self.layer1(n))
        n = nnFunc.relu(self.layer2(n))
        n = self.out(n)
        return n


cnnModel = CNNModel()

# https://pytorch.org/docs/master/optim.html
optim = torch.optim.Adam(cnnModel.parameters(), lr=lRate)

criterion = nn.MSELoss()
epoch=numEpoch
trainLossSet = []

for i in range(epoch):
    trainLoss=0
    # print('Epoch: '+str(i))
    for xVal, yVal in zip(x_train, y_train):
        
        # Sets the gradients of all optimized torch.Tensor s to zero. 
        optim.zero_grad()

        output = cnnModel(xVal)
        loss = criterion(output, yVal.unsqueeze(0))
        trainLoss += loss
        loss.backward()

        # Performs a single optimization step.
        optim.step()

    trainLossSet.append(trainLoss.item())

print("trainLoss change over Epochs= ", trainLossSet)

testLoss = 0
y_pred=[]
for xt, yt in zip(x_test, y_test):
    out = cnnModel(xt)
    y_pred.append(out.item())
    if(debug):
        print(out)
        print(out, yt)
    loss = criterion(out, yt.unsqueeze(0))
    if(debug):
        print('test'+str(loss))
    testLoss += loss

r2 = r2_score(y_test, y_pred)
testLoss_avg = testLoss / len(x_test)
trainLoss_avg = trainLoss / len(x_train)
print("Average Training Loss : "+ str(trainLoss_avg.item()))
print("Average Test Loss : "+ str(testLoss_avg.item()))
print("MeanSqErr = ", str(testLoss_avg.item()), "R2 = ", r2)


#predictor = LinearRegression(n_jobs=-1)
#predictor.fit(X=TRAIN_INPUT, y=TRAIN_OUTPUT)
#outcome = predictor.predict(X=X_TEST)
#coefficients = predictor.coef_
#print('Outcome : {}\nCoefficients : {}'.format(outcome, coefficients))

#######################################################################kernel=

print("\nSVR regression -- linear:")
regressorL = SVR(kernel='linear')
regressorL.fit(X=x_train, y=y_train)
y_pred = regressorL.predict(X=x_test)
mse = mean_squared_error(y_pred, y_test)
r2 = r2_score(y_test, y_pred)
print("MeanSqErr Linear = ", mse, "R2 = ", r2)
if(debug):
    print(y_pred)
    print(x_test, y_pred)

################################################
print("\n SVR regression -- polynomial:")

mse = 0
r2 = 0
regressorP = SVR(kernel='poly')
regressorP.fit(X=x_train, y=y_train)
y_pred = regressorP.predict(X=x_test)
mse = mean_squared_error(y_pred, y_test)
r2 = r2_score(y_test, y_pred)
print("MeanSqErr Poly = ", mse, "R2 = ", r2)
if(debug):
    print(y_pred)

################################################
print("\nSVR regression -- rbf:")

mse = 0
r2 = 0
regressor = SVR(kernel='rbf')
regressor.fit(X=x_train, y=y_train)
y_pred = regressor.predict(X=x_test)
mse = mean_squared_error(y_pred, y_test)
r2 = r2_score(y_test, y_pred)
print("MeanSqErr RBF = ", mse, "R2 = ", r2)
if(debug):
    print(y_pred)

####################################################3

numLayers = nStage
numIter = 99
# Multi-layer Perceptron activation{‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, default=’relu’
print("\nANN-identity with learning rate:", numLayers)

regressor = MLPRegressor(activation='identity',hidden_layer_sizes=numLayers,max_iter=numIter)
regressor.fit(X=x_train, y=y_train)
y_pred = regressor.predict(X=x_test)
mse = mean_squared_error(y_pred, y_test)
r2 = r2_score(y_test, y_pred)
print("MeanSqErr Test= ", mse, "R2 = ", r2)
print("MSE Train = ", regressor.loss_curve_)
mse = regressor.loss_curve_
if(debug):
    print(y_pred, mse[len(mse)-1])
    print("MSE= ", mse[len(mse)-1])

####################################################3

print("\nANN-logistic with learning rate:", numLayers)
regressor = MLPRegressor(activation='logistic',hidden_layer_sizes=numLayers,max_iter=numIter)
regressor.fit(X=x_train, y=y_train)
y_pred = regressor.predict(X=x_test)
mse = mean_squared_error(y_pred, y_test)
r2 = r2_score(y_test, y_pred)
print("MeanSqErr Test= ", mse, "R2 = ", r2)
print("MSE Train = ", regressor.loss_curve_)
if(debug):
    print(regressor.loss_curve_)
    print(y_pred, mse[len(mse)-1])
    print(mse[len(mse)-1])

####################################################3
