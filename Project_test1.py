import numpy
import random
import math

class Network:
    def __init__(self, no_layers, no_neurons, no_initial_inputs):
        self.layers = int(no_layers)
        self.neurons = int(no_neurons)
        self.no_ini_inputs = int(no_initial_inputs)

    #inp_act needs to be inputted as a list of all of the input activations. this can be assigned from the first input or just the input
    #into that layer later on when actually being coded.

    def weight_set(self, inp_act):
        input_weights = []
        weight = []
        for x in range(0, self.neurons - 1):
            for i in range(0, len(inp_act)-1):
                input_weights[i] = random.randint(-1,1)
            weight[x] = input_weights
            input_weights = []
        return weight

    def bias_set(self):
        bias = []
        for i in range(0,self.neurons - 1):
            x = random.randint(-1,1)
            bias.append(int(x))
        return bias

    def z(self, inp_act, weights, bias):
        z = []
        for x in range(0, len(weights)):
            for i in range(0, len(weights[1])):
                temp += inp_act[i]*weight[x][i]
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









