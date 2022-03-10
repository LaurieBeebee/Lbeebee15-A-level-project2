import csv
import random
import pandas as pd


def database_test():
    Iris = open("Iris.csv", "r")
    Irisreader = csv.reader(Iris)
    for row in Irisreader:
        if row[0] == "50":
            print(row)


def weight_set_test():
    input_weights = []
    weights = []
    for y in range(0, 4):
        for x in range(0, 4):
            # z = round(random.uniform(-1,1),2)
            print(input_weights)
            input_weights.append(2)
        weights[y] = input_weights
        input_weights = []
    return weights


print(pd.read_csv('Iris.csv'))
x = pd.read_csv('Iris.csv')
print(x.loc[:150, ['SepalLengthCm', 'PetalLengthCm']])