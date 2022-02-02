import numpy
import random
import csv


class Network:
    def __init__(self, no_hidden_layers, no_hidden_layer_neurons, no_outputs, learning_rate):
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
        self.learning_rate = int(learning_rate)
        if self.correct_output == "Iris-setosa": #would need to be changed if using different database
            self.desired_outputs = [1,0,0]
        elif self.correct_output == "Iris-versicolor":
            self.desired_outputs = [0,1,0]
        elif self.correct_output == "Iris-virginica":
            self.desired_outputs = [0,0,1]


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

    def inv_sigmoid(self,x):
        x = float(x)
        return (numpy.exp(-x)/(numpy.exp(-x)+1)**2)

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
        for i in range(len(self.desired_outputs)):
            cost.append(0.5*((self.desired_outputs[i] - activations[self.layers][i])**2))
        return cost

    def final_layer_error(self, z, activations):
        final_layer_error = []
        for i in range(self.no_outputs):
            final_layer_error.append(((activations[self.layers][i]-self.desired_outputs[i])*self.inv_sigmoid(z[self.layers][i])))
        return final_layer_error

    def dicttomat_weights(self,weights,x): # have to remember that this is individual to each layer and have to know how many neurons in each layer and change it each time.
        mat = weights[x]
        mat = numpy.asarray(mat)
        return(mat)

    def errorbackprop(self,weights,final_layer_error,z):
        layer_error = []
        all_layer_error = []
        for i in reversed(range(1,len(weights))):
            mat = self.dicttomat_weights(weights,i)
            if i == len(weights)-1:
                layer_error = numpy.reshape(final_layer_error,((len(final_layer_error)),1))
            for x in range(0, len(z[i-1])):
                z[i-1][x] = self.inv_sigmoid(z[i-1][x])
            z[i-1] = numpy.reshape(z[i-1],(len(z[i-1]),1))
            prev_layer_error = numpy.multiply((numpy.matmul(mat.transpose(),layer_error)),(z[i-1]))
            all_layer_error.append(prev_layer_error)
            layer_error = prev_layer_error
        all_layer_error.append(numpy.reshape(final_layer_error,(len(final_layer_error),1)))
        return(all_layer_error)

    def derivcosttoweight(self,activations,first_layer_activations,errors):
        count = 0
        derive_cost_to_weight = {}
        temp = []
        input_temp = []
        activ = first_layer_activations
        for i in range(0, len(errors)):
            for x in range(0, len(errors[i])):
                for y in range(0, len(activ)):
                    temp.append(activ[y]*errors[i][x][0])
                input_temp.append(temp)
                temp = []
            derive_cost_to_weight[count] = input_temp
            input_temp = []
            activ = activations[count]
            count+=1
        return(derive_cost_to_weight)

    def gradient_descent(self, weights, biases, derivcosttoweight, errors):
        for i in range(0, len(weights)):
            for x in range(0, len(weights[i])):
                for y in range(0, len(weights[i][x])):
                    weights[i][x][y] += self.learning_rate*derivcosttoweight[i][x][y]
        for i in range(0,len(biases)):
            for x in range(0, len(biases[i])):
                biases[i][x] += self.learning_rate*errors[i][x][0]
        return(weights, biases)

    def output(self,activations):
        output = ""
        if activations[self.layers][0] > activations[self.layers][1] and activations[self.layers][0] > activations[self.layers][2]:
            output = "Iris-setosa"
        elif activations[self.layers][1] > activations[self.layers][0] and activations[self.layers][1] > activations[self.layers][2]:
            output = "Iris-versicolor"
        else:
            output = "Iris-virginica"
        return output


    def main(self):
        count = 1
        weights = self.total_weights()
        biases = self.total_bias()
        while count <= 70 and count != 0:
            count = 1
            for i in range(0,99):
                z = self.z(weights,biases)
                activations = self.activations(z)
                first_layer_activations = self.first_layer_activations()
                cost = self.cost(activations)
                final_layer_error = self.final_layer_error(z,activations)
                all_error = self.errorbackprop(weights,final_layer_error,z)
                deriv_cost_to_weight = self.derivcosttoweight(activations,first_layer_activations,all_error)
                weights = self.gradient_descent(weights,biases,deriv_cost_to_weight,all_error)[0]
                biases = self.gradient_descent(weights,biases,deriv_cost_to_weight,all_error)[1]
                if self.output(activations) == self.correct_output:
                    count+=1
                Iris = open("Iris.csv", "r")
                Irisreader = csv.reader(Iris)
                self.test_data_no = random.randint(1, 150)
                test_data = []
                for row in Irisreader:  # can make this a lot better by doing a binary search rather than just a linear search
                    if row[0] == str(self.test_data_no):
                        test_data = [row[1], row[2], row[3], row[4]]  # this is specific to this database so if changing the code this must be edited.
                        self.correct_output = str(row[5])
                Iris.close()
                self.inp_act = test_data
            print(count)









flower = Network(2,5,3,3)
# all_weights = flower.total_weights()
# print(all_weights)
# all_biases = flower.total_bias()
# print(all_biases)
# all_z = flower.z(all_weights,all_biases)
# print(all_z)
# activations = flower.activations(all_z)
# print(activations)
# first_layer_activations = flower.first_layer_activations()
# print(first_layer_activations)
# cost = flower.cost(activations)
# print(cost)
# print(flower.correct_output)
# final_layer_error = flower.final_layer_error(all_z,activations)
# print(final_layer_error)
# layer_1_matrix = flower.dicttomat_weights(all_weights,0) # have to remember that this is individual to each layer and have to change each time
# print(layer_1_matrix)
# all_error = flower.errorbackprop(all_weights,final_layer_error, all_z)
# print(all_error)
# derivcosttoweight = flower.derivcosttoweight(activations,first_layer_activations,all_error)
# print(derivcosttoweight)
# gradient_descent = flower.gradient_descent(all_weights,all_biases,derivcosttoweight,all_error)
# print(gradient_descent[0])
# print(gradient_descent[1])
flower.main()








