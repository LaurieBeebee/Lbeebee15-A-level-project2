import csv
import random


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


weight_set_test()
