import numpy
import random
import csv


class Network:
    def __init__(self, no_layers, no_neurons, input_activations, no_outputs):
        self.layers = int(no_layers)
        self.neurons = int(no_neurons)
        self.inp_act = input_activations
        self.no_outputs = int(no_outputs)

    #inp_act needs to be inputted as a list of all of the input activations. this can be assigned from the first input or just the input
    #into that layer later on when actually being coded.

    #got database now but need to think about how im going to loop through the number of neurons in each layer when setting weights
    #biases as it is changing and may not want same number of neurons in each layer. also need to think about final layer and how that
    #will work

    def weight_set(self, inp_act, no_neurons):
        input_weights = []
        weight = []
        for x in range(0, no_neurons):
            for i in range(0, len(inp_act)):
                rand = random.uniform(-1,1)
                input_weights.append(rand)
            weight.append(input_weights)
            input_weights = []
        return weight

    def bias_set(self):
        bias = []
        for i in range(0,self.neurons):
            x = random.uniform(-1,1)
            bias.append(int(x))
        return bias

    def z(self, inp_act, weights, bias):
        z = []
        for x in range(0, len(weights)):
            for i in range(0, len(weights[1])):
                temp += inp_act[i]*weights[x][i]
            z.append(temp + bias[z])
            temp = 0
        return z

    def sigmoid(self,x):
        return 1/(1+numpy.exp(-x))


    def activations(self,z):
        act = []
        for i in range(0,self.neurons - 1):
            act[i] = self.sigmoid(z[i])
        return act

    def total_weights(self, inp_act):
        weights = {}
        input_weights = inp_act
        for i in range(0, self.layers):
            weights[i] = self.weight_set(input_weights, self.neurons)
            input_weights = weights[i]
            if i+1 == self.layers:
                weights[i+1] = self.weight_set(input_weights, self.no_outputs)
        return weights





Iris = open("Iris.csv","r")
Irisreader = csv.reader(Iris)
test_data_no = random.randint(1,150)
test_data = []
for row in Irisreader:
    if row[0] == str(test_data_no):
        test_data = [row[1],row[2],row[3],row[4]] #this is specific to this database so if changing the code this must be edited.
flower = Network(2,5,test_data,3)
all_weights = flower.total_weights(test_data)

print(all_weights)












