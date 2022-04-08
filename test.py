import csv
import random
import pandas as pd
import numpy as np


# def database_test():
#     Iris = open("Iris.csv", "r")
#     Irisreader = csv.reader(Iris)
#     for row in Irisreader:
#         if row[0] == "50":
#             print(row)
#
#
# def weight_set_test():
#     input_weights = []
#     weights = []
#     for y in range(0, 4):
#         for x in range(0, 4):
#             # z = round(random.uniform(-1,1),2)
#             print(input_weights)
#             input_weights.append(2)
#         weights[y] = input_weights
#         input_weights = []
#     return weights
#
#
# print(pd.read_csv('Iris.csv'))
# x = pd.read_csv('Iris.csv')
# print(x.loc[:150, ['SepalLengthCm', 'PetalLengthCm']])

#testing the data augmentation for the database

x = pd.read_csv('Iris.csv')
X = x.loc[:150, ["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]
print(X)
X = X.values
print(X)
print(X.transpose())

# #X[len(X)+1] += [5.2,5.2,5.2,5.2]
# #X = np.append(Y,[[5.2,5.2,5.2,5.2]], axis=0)
# for i in range(150):
#     temp = X[i].copy()
#     no = random.randint(0,3)
#     temp[no] = temp[no] * 0.999
#     X = np.append(X, [temp], axis = 0)
#
# np.set_printoptions(threshold = 3)
# print(X)


# print(np.zeros((4,5)))



