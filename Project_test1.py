import numpy
import random
import csv


class Network:
    def __init__(self, no_hidden_layers, no_hidden_layer_neurons, no_outputs):
        Iris = open("Iris.csv", "r")
        Irisreader = csv.reader(Iris)
        self.test_data_no = random.randint(1, 150)
        test_data = []
        for row in Irisreader:  # can make this a lot better by doing a binary search rather than just a linear search
            if row[0] == str(self.test_data_no):
                test_data = [row[1], row[2], row[3], row[4]]# this is specific to this database so if changing the code this must be edited.
                self.correct_output = str(row[5])
        Iris.close()
        self.layers = int(no_hidden_layers)
        self.neurons = int(no_hidden_layer_neurons)
        self.inp_act = test_data
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

    def bias_set(self, no_neurons):
        bias = []
        for i in range(0,no_neurons):
            x = random.uniform(-1,1)
            bias.append(x)
        return bias

    def total_weights(self):
        weights = {}
        input_weights = self.inp_act
        for i in range(0, self.layers):
            weights[i] = self.weight_set(input_weights, self.neurons)
            input_weights = weights[i]
            if i+1 == self.layers:
                weights[i+1] = self.weight_set(input_weights, self.no_outputs)
        return weights

    def total_bias(self):
        biases = {}
        for i in range(0, self.layers):
            biases[i] = self.bias_set(self.neurons)
            if i+1 == self.layers:
                biases[i+1] = self.bias_set((self.no_outputs))
        return biases

    def sigmoid(self,x):
        x = float(x)
        return 1/(1+numpy.exp(-x))

    def first_layer_activations(self):
        activations = []
        for i in range(0,len(self.inp_act)):
            activations.append(self.sigmoid(self.inp_act[i]))
        return activations

    def z(self, weights, biases):
        z = {}
        temp = []
        z_input = []
        temp_sum = 0
        prev_activ = self.first_layer_activations()
        for i in range(0,len(weights)):
            for x in range(0,len(weights[i])):
                for y in range(0,len(weights[i][x])):
                    temp.append(weights[i][x][y]*prev_activ[y]) #have no clue if this is the right number of loops
                    if y+1 == len(weights[i][x]):
                        for h in range(0,len(temp)):
                            temp_sum += temp[h]
                        temp_sum += biases[i][x]
                        temp = []
                z_input.append(temp_sum)
                temp_sum = 0
            prev_activ = []
            for q in range(0,len(z_input)):  # i think i fixed the error by removing the -1 here
                prev_activ.append(self.sigmoid(z_input[q]))
            z[i] = z_input
            z_input = []
        return z

    def activations(self, z):
        activations = {}
        temp_activations = []
        for i in range(0, len(z)):
            for x in range(0, len(z[i])):
                temp_activations.append(self.sigmoid(z[i][x]))
            activations[i] = temp_activations
            temp_activations = []
        return activations

    def cost(self, activations):
        cost = []
        cost_temp = 0
        if self.correct_output == "Iris-setosa":
            cost_temp = 0.5*((1 - activations[self.layers][0])**2)
            cost.append(cost_temp)
            cost_temp = 0.5*((0 - activations[self.layers][1])**2)
            cost.append(cost_temp)
            cost_temp = 0.5*((0 - activations[self.layers][2])**2)
            cost.append(cost_temp)
        elif self.correct_output == "Iris-versicolor":
            cost_temp = 0.5*((0 - activations[self.layers][0])**2)
            cost.append(cost_temp)
            cost_temp = 0.5*((1 - activations[self.layers][1])**2)
            cost.append(cost_temp)
            cost_temp = 0.5*((0 - activations[self.layers][2])**2)
            cost.append(cost_temp)
        elif self.correct_output == "Iris-virginica":
            cost_temp = 0.5*((0 - activations[self.layers][0])**2)
            cost.append(cost_temp)
            cost_temp = 0.5*((0 - activations[self.layers][1])**2)
            cost.append(cost_temp)
            cost_temp = 0.5*((1 - activations[self.layers][2])**2)
            cost.append(cost_temp)
        return cost




flower = Network(2,5,3)
all_weights = flower.total_weights()
print(all_weights)
all_biases = flower.total_bias()
print(all_biases)
all_z = flower.z(all_weights,all_biases)
print(all_z)
activations = flower.activations(all_z)
print(activations)
cost = flower.cost(activations)
print(cost)
print(flower.correct_output)










