import csv

Iris = open("Iris.csv","r")
Irisreader = csv.reader(Iris)
for row in Irisreader:
    if row[0] == "50":
        print(row)



