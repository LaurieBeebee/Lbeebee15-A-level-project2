import numpy as np
import csv
import time
import pandas as pd
from sklearn.model_selection import train_test_split
#import mapplotlib.pyplot as plt

# 2 hidden layers, 5 neurons each

LEARNING_RATE = 0.005
#
# X = np.zeros((4, 130), float)
Y = np.zeros((3, 150), int)

Iris = pd.read_csv("Iris.csv")
X = Iris.loc[:150, ["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]
y = Iris.loc[:150, ["Species"]]
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train = np.transpose(X_train.values)
X_test = np.transpose(X_test.values)
Y_train = np.transpose(Y_train.values)
Y_test = np.transpose(Y_test.values)

shape = Y_train.shape
Y = np.zeros((3, shape[1]), int)
for i in range(0, shape[1]):
    if Y_train[0, i] == "Iris-setosa":
        Y[0, i] = 1
        Y[1, i] = 0
        Y[2, i] = 0
    elif Y_train[0, i] == "Iris-versicolor":
        Y[0, i] = 0
        Y[1, i] = 1
        Y[2, i] = 0
    elif Y_train[0, i] == "Iris-virginica":
        Y[0, i] = 0
        Y[1, i] = 0
        Y[2, i] = 1



# count = 0
# for row in Irisreader:
#     if (row[0] >= "1" and row[0] < "41") or (row[0] >= "51" and row[0] < "91") or (row[0] >= "101" and row[0] < "141"):
#         X[0, count] = float(row[1])
#         X[1, count] = float(row[2])
#         X[2, count] = float(row[3])
#         X[3, count] = float(row[4])
#         if row[5] == "Iris-setosa":
#             Y[0, count] = 1
#             Y[1, count] = 0
#             Y[2, count] = 0
#         elif row[5] == "Iris-versicolor":
#             Y[0, count] = 0
#             Y[1, count] = 1
#             Y[2, count] = 0
#         elif row[5] == "Iris-virginica":
#             Y[0, count] = 0
#             Y[1, count] = 0
#             Y[2, count] = 1
#         count += 1
# Iris.close()
#

def weight_set():
    w0 = np.random.randn(5, 4)
    w1 = np.random.randn(5, 5)
    w2 = np.random.randn(3, 5)
    return w0, w1, w2

def bias_set():
    b1 = np.zeros((5,1))
    b2 = np.zeros((5,1))
    b3 = np.zeros((3,1))
    return b1, b2, b3

def sigmoid(x):
        x = x.astype(float)
        return 1/(1+np.exp(-x))

def inv_sigmoid(x):
        x = x.astype(float)
        return (np.exp(-x)/(np.exp(-x)+1)**2)

def z(weight, prev_activation, bias):
    z = np.dot(weight, prev_activation) + bias
    return z

def all_z_activations(w0, w1, w2, b1, b2, b3, x):
    z1 = z(w0, x, b1)
    a1 = sigmoid(z1)
    z2 = z(w1, a1, b2)
    a2 = sigmoid(z2)
    z3 = z(w2, a2, b3)
    a3 = sigmoid(z3)
    return z1, z2, z3, a1, a2, a3

def errors(w1, w2, z1, z2, z3, a3, Y):
    errors3 = (a3 - Y) #* inv_sigmoid(z3)
    errors2 = np.dot(np.transpose(w2), errors3)*inv_sigmoid(z2)
    errors1 = np.dot(np.transpose(w1), errors2)*inv_sigmoid(z1)
    return errors1, errors2, errors3

def deriv_cost_to_weight(errors1, errors2, errors3, a1, a2, x):
    dw0 = 1/np.shape(x)[1]*(np.dot(errors1, np.transpose(x)))
    dw1 = 1/np.shape(x)[1]*(np.dot(errors2, np.transpose(a1)))
    dw2 = 1/np.shape(x)[1]*(np.dot(errors3, np.transpose(a2)))
    return dw0, dw1, dw2

def gradient_descent(w0, w1, w2, b1, b2, b3, error1, error2, error3, dw0, dw1, dw2, x):
    w0 -= LEARNING_RATE*dw0
    w1 -= LEARNING_RATE*dw1
    w2 -= LEARNING_RATE*dw2
    b1 -= LEARNING_RATE*(1/np.shape(x)[1])*np.sum(error1, axis=1, keepdims=True)
    b2 -= LEARNING_RATE*(1/np.shape(x)[1])*np.sum(error2, axis=1, keepdims=True)
    b3 -= LEARNING_RATE*(1/np.shape(x)[1])*np.sum(error3, axis=1, keepdims=True)
    return w0, w1, w2, b1, b2, b3

def cost(Y, a3):
    #cost = 1/np.shape(Y)[1]*np.sum((Y - a3)**2)
    #cost = np.dot(Y, np.log(np.transpose(a3))) + np.dot((1 - Y), np.log(np.transpose(1 - a3)))
    cost = (-1./np.shape(Y)[1]) * np.sum(Y * np.log(a3) + (1 - Y) * np.log(1 - a3))
    return cost


def main_train():
    costs = [] #keep track of costs
    tic = time.time()
    weights = weight_set()
    biases = bias_set()
    z_and_activations = all_z_activations(weights[0], weights[1], weights[2], biases[0], biases[1], biases[2], X_train)
    toc = time.time()
    while toc-tic <= 50:
        z_and_activations = all_z_activations(weights[0], weights[1], weights[2], biases[0], biases[1], biases[2], X_train)
        error = errors(weights[1], weights[2], z_and_activations[0], z_and_activations[1], z_and_activations[2], z_and_activations[5], Y)
        derivitive_cost_to_weight = deriv_cost_to_weight(error[0], error[1], error[2], z_and_activations[3], z_and_activations[4], X_train)
        change_weights_and_biases = gradient_descent(weights[0], weights[1], weights[2], biases[0], biases[1], biases[2], error[0], error[1], error[2], derivitive_cost_to_weight[0], derivitive_cost_to_weight[1], derivitive_cost_to_weight[2], X_train)
        cost1 = cost(Y, z_and_activations[5])
        print(cost1)
        costs.append(cost1)
        toc = time.time()
    count = 0
    test_result = 0
    while count != 30:
        test_result += test(weights[0], weights[1], weights[2], biases[0], biases[1], biases[2])
        count += 1
    print(test_result/count)

def test(w0, w1, w2, b1, b2, b3):
    # X = np.zeros((4, 150), float)
    # Y = np.zeros((3, 150), int)
    #
    correct = 0
    #
    # Iris = open("Iris.csv", "r")
    # Irisreader = csv.reader(Iris)
    # count = 0
    # for row in Irisreader:
    #     if row[0] != "Id":
    #         #if (row[0] >= "41" and row[0] < "51") or (row[0] >= "91" and row[0] < "101") or (row[0] >= "141" and row[0] < "151"):
    #             X[0, count] = float(row[1])
    #             X[1, count] = float(row[2])
    #             X[2, count] = float(row[3])
    #             X[3, count] = float(row[4])
    #             if row[5] == "Iris-setosa":
    #                 Y[0, count] = 1
    #                 Y[1, count] = 0
    #                 Y[2, count] = 0
    #             elif row[5] == "Iris-versicolor":
    #                 Y[0, count] = 0
    #                 Y[1, count] = 1
    #                 Y[2, count] = 0
    #             elif row[5] == "Iris-virginica":
    #                 Y[0, count] = 0
    #                 Y[1, count] = 0
    #                 Y[2, count] = 1
    #             count += 1
    # Iris.close()
    shape = Y_test.shape
    Y = np.zeros((3, shape[1]), int)
    for i in range(0, shape[1]):
        if Y_test[0, i] == "Iris-setosa":
            Y[0, i] = 1
            Y[1, i] = 0
            Y[2, i] = 0
        elif Y_test[0, i] == "Iris-versicolor":
            Y[0, i] = 0
            Y[1, i] = 1
            Y[2, i] = 0
        elif Y_test[0, i] == "Iris-virginica":
            Y[0, i] = 0
            Y[1, i] = 0
            Y[2, i] = 1


    activations = all_z_activations(w0, w1, w2, b1, b2, b3, X_test)
    for i in range(0, np.shape(X_test)[1]):
        if activations[5][0,i] > activations[5][1,i] and activations[5][0,i] > activations[5][2,i]:
            output = "Iris-setosa"
            if Y[0,i] == 1:
                correct += 1
        elif activations[5][1,i] > activations[5][0,i] and activations[5][1,i] > activations[5][2,i]:
            output = "Iris-versicolor"
            if Y[1,i] == 1:
                correct += 1
        elif activations[5][2,i] > activations[5][1,i] and activations[5][2,i] > activations[5][0,i]:
            output = "Iris-virginica"
            if Y[2,i] == 1:
                correct += 1
    return (correct/(np.shape(X_test)[1]))*100






main_train()