import numpy as np
import csv
import time


# 2 hidden layers, 5 neurons each

LEARNING_RATE = 0.005

X = np.zeros((4, 130), float)
Y = np.zeros((3, 130), int)

Iris = open("Iris.csv", "r")
Irisreader = csv.reader(Iris)
count = 0
for row in Irisreader:
    if (row[0] >= "1" and row[0] < "41") or (row[0] >= "51" and row[0] < "91") or (row[0] >= "101" and row[0] < "141"):
        X[0, count] = float(row[1])
        X[1, count] = float(row[2])
        X[2, count] = float(row[3])
        X[3, count] = float(row[4])
        if row[5] == "Iris-setosa":
            Y[0, count] = 1
            Y[1, count] = 0
            Y[2, count] = 0
        elif row[5] == "Iris-versicolor":
            Y[0, count] = 0
            Y[1, count] = 1
            Y[2, count] = 0
        elif row[5] == "Iris-virginica":
            Y[0, count] = 0
            Y[1, count] = 0
            Y[2, count] = 1
        count += 1
Iris.close()

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
    cost = np.dot(Y, (np.log(a3)) + np.dot((1 - Y)), np.log(1 - a3))
    return cost

def main():
    tic = time.time()
    weights = weight_set()
    biases = bias_set()
    z_and_activations = all_z_activations(weights[0], weights[1], weights[2], biases[0], biases[1], biases[2], X)
    toc = time.time()
    while cost(Y, z_and_activations[5]) > 0 and toc-tic <= 50:
        z_and_activations = all_z_activations(weights[0], weights[1], weights[2], biases[0], biases[1], biases[2], X)
        error = errors(weights[1], weights[2], z_and_activations[0], z_and_activations[1], z_and_activations[2], z_and_activations[5], Y)
        derivitive_cost_to_weight = deriv_cost_to_weight(error[0], error[1], error[2], z_and_activations[3], z_and_activations[4], X)
        change_weights_and_biases = gradient_descent(weights[0], weights[1], weights[2], biases[0], biases[1], biases[2], error[0], error[1], error[2], derivitive_cost_to_weight[0], derivitive_cost_to_weight[1], derivitive_cost_to_weight[2], X)
        cost1= cost(Y, z_and_activations[5])
        print(cost1)
        toc = time.time()
    count = 0
    test_result = 0
    while count != 30:
        test_result += test(weights[0], weights[1], weights[2], biases[0], biases[1], biases[2])
        count += 1
    print(test_result/count)

def test(w0, w1, w2, b1, b2, b3):
    X = np.zeros((4, 150), float)
    Y = np.zeros((3, 150), int)

    correct = 0

    Iris = open("Iris.csv", "r")
    Irisreader = csv.reader(Iris)
    count = 0
    for row in Irisreader:
        if row[0] != "Id":
            #if (row[0] >= "41" and row[0] < "51") or (row[0] >= "91" and row[0] < "101") or (row[0] >= "141" and row[0] < "151"):
                X[0, count] = float(row[1])
                X[1, count] = float(row[2])
                X[2, count] = float(row[3])
                X[3, count] = float(row[4])
                if row[5] == "Iris-setosa":
                    Y[0, count] = 1
                    Y[1, count] = 0
                    Y[2, count] = 0
                elif row[5] == "Iris-versicolor":
                    Y[0, count] = 0
                    Y[1, count] = 1
                    Y[2, count] = 0
                elif row[5] == "Iris-virginica":
                    Y[0, count] = 0
                    Y[1, count] = 0
                    Y[2, count] = 1
                count += 1
    Iris.close()
    activations = all_z_activations(w0, w1, w2, b1, b2, b3, X)
    for i in range(0, np.shape(X)[1]):
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
    return (correct/count)*100






main()

# weights = weight_set()
# biases = bias_set()
# z_and_activations = all_z_activations(weights[0], weights[1], weights[2], biases[0], biases[1], biases[2], X)
# errors = errors(weights[1], weights[2], z_and_activations[0], z_and_activations[1], z_and_activations[2], z_and_activations[5], Y)
# deriv_cost_to_weight = deriv_cost_to_weight(errors[0], errors[1], errors[2], z_and_activations[3], z_and_activations[4], X)
# gradient_descent = gradient_descent(weights[0], weights[1], weights[2], biases[0], biases[1], biases[2], errors[0], errors[1], errors[2], deriv_cost_to_weight[0], deriv_cost_to_weight[1], deriv_cost_to_weight[2], X)
# cost_1 = cost(Y, z_and_activations[5])
# print(cost_1)

#
# class Network:
#     def __init__(self, no_hidden_layers, no_hidden_layer_neurons, no_outputs, learning_rate):
#         Iris = open("Iris.csv", "r")
#         Irisreader = csv.reader(Iris)
#         self.test_data_no = random.randint(1, 150)
#         test_data = []
#         for row in Irisreader:  # can make this a lot better by doing a binary search rather than just a linear search
#             if row[0] == str(self.test_data_no):
#                 test_data = [row[1], row[2], row[3], row[4]]# this is specific to this database so if changing the code this must be edited.
#                 self.correct_output = str(row[5])
#         Iris.close()
#         self.layers = int(no_hidden_layers)
#         self.neurons = int(no_hidden_layer_neurons)
#         self.inp_act = test_data
#         self.no_outputs = int(no_outputs)
#         self.learning_rate = int(learning_rate)
#         if self.correct_output == "Iris-setosa": #would need to be changed if using different database
#             self.desired_outputs = [1,0,0]
#         elif self.correct_output == "Iris-versicolor":
#             self.desired_outputs = [0,1,0]
#         elif self.correct_output == "Iris-virginica":
#             self.desired_outputs = [0,0,1]
#
#
#     #inp_act needs to be inputted as a list of all of the input activations. this can be assigned from the first input or just the input
#     #into that layer later on when actually being coded.
#
#     #got database now but need to think about how im going to loop through the number of neurons in each layer when setting weights
#     #biases as it is changing and may not want same number of neurons in each layer. also need to think about final layer and how that
#     #will work
#
#     def weight_set(self, inp_act, no_neurons):
#         input_weights = []
#         weight = []
#         for x in range(0, no_neurons):
#             for i in range(0, len(inp_act)):
#                 rand = random.uniform(-1,1)
#                 input_weights.append(rand)
#             weight.append(input_weights)
#             input_weights = []
#         return weight
#
#     def bias_set(self, no_neurons):
#         bias = []
#         for i in range(0,no_neurons):
#             x = random.uniform(-1,1)
#             bias.append(x)
#         return bias
#
#     def total_weights(self):
#         weights = {}
#         input_weights = self.inp_act
#         for i in range(0, self.layers):
#             weights[i] = self.weight_set(input_weights, self.neurons)
#             input_weights = weights[i]
#             if i+1 == self.layers:
#                 weights[i+1] = self.weight_set(input_weights, self.no_outputs)
#         return weights
#
#     def total_bias(self):
#         biases = {}
#         for i in range(0, self.layers):
#             biases[i] = self.bias_set(self.neurons)
#             if i+1 == self.layers:
#                 biases[i+1] = self.bias_set((self.no_outputs))
#         return biases
#
#     def sigmoid(self,x):
#         x = float(x)
#         return 1/(1+np.exp(-x))
#
#     def inv_sigmoid(self,x):
#         x = float(x)
#         return (np.exp(-x)/(np.exp(-x)+1)**2)
#
#     def first_layer_activations(self):
#         activations = []
#         for i in range(0,len(self.inp_act)):
#             activations.append(self.sigmoid(self.inp_act[i]))
#         return activations
#
#     def z(self, weights, biases):
#         z = {}
#         temp = []
#         z_input = []
#         temp_sum = 0
#         prev_activ = self.first_layer_activations()
#         for i in range(0,len(weights)):
#             for x in range(0,len(weights[i])):
#                 for y in range(0,len(weights[i][x])):
#                     temp.append(weights[i][x][y]*prev_activ[y]) #have no clue if this is the right number of loops
#                     if y+1 == len(weights[i][x]):
#                         for h in range(0,len(temp)):
#                             temp_sum += temp[h]
#                         temp_sum += biases[i][x]
#                         temp = []
#                 z_input.append(temp_sum)
#                 temp_sum = 0
#             prev_activ = []
#             for q in range(0,len(z_input)):  # i think i fixed the error by removing the -1 here
#                 prev_activ.append(self.sigmoid(z_input[q]))
#             z[i] = z_input
#             z_input = []
#         return z
#
#     def activations(self, z):
#         activations = {}
#         temp_activations = []
#         for i in range(0, len(z)):
#             for x in range(0, len(z[i])):
#                 temp_activations.append(self.sigmoid(z[i][x]))
#             activations[i] = temp_activations
#             temp_activations = []
#         return activations
#
#     def cost(self, activations):
#         cost = []
#         for i in range(len(self.desired_outputs)):
#             cost.append(0.5*((self.desired_outputs[i] - activations[self.layers][i])**2))
#         return cost
#
#     def final_layer_error(self, z, activations):
#         final_layer_error = []
#         for i in range(self.no_outputs):
#             final_layer_error.append(((activations[self.layers][i]-self.desired_outputs[i])*self.inv_sigmoid(z[self.layers][i])))
#         return final_layer_error
#
#     def dicttomat_weights(self,weights,x): # have to remember that this is individual to each layer and have to know how many neurons in each layer and change it each time.
#         mat = weights[x]
#         mat = np.asarray(mat)
#         return(mat)
#
#     def errorbackprop(self,weights,final_layer_error,z):
#         layer_error = []
#         all_layer_error = []
#         for i in reversed(range(1,len(weights))):
#             mat = self.dicttomat_weights(weights,i)
#             if i == len(weights)-1:
#                 layer_error = np.reshape(final_layer_error,((len(final_layer_error)),1))
#             for x in range(0, len(z[i-1])):
#                 z[i-1][x] = self.inv_sigmoid(z[i-1][x])
#             z[i-1] = np.reshape(z[i-1],(len(z[i-1]),1))
#             prev_layer_error = np.multiply((np.matmul(mat.transpose(),layer_error)),(z[i-1]))
#             all_layer_error.append(prev_layer_error)
#             layer_error = prev_layer_error
#         all_layer_error.append(np.reshape(final_layer_error,(len(final_layer_error),1)))
#         return(all_layer_error)
#
#     def derivcosttoweight(self,activations,first_layer_activations,errors):
#         count = 0
#         derive_cost_to_weight = {}
#         temp = []
#         input_temp = []
#         activ = first_layer_activations
#         for i in range(0, len(errors)):
#             for x in range(0, len(errors[i])):
#                 for y in range(0, len(activ)):
#                     temp.append(activ[y]*errors[i][x][0])
#                 input_temp.append(temp)
#                 temp = []
#             derive_cost_to_weight[count] = input_temp
#             input_temp = []
#             activ = activations[count]
#             count+=1
#         return(derive_cost_to_weight)
#
#     def gradient_descent(self, weights, biases, derivcosttoweight, errors):
#         for i in range(0, len(weights)):
#             for x in range(0, len(weights[i])):
#                 for y in range(0, len(weights[i][x])):
#                     weights[i][x][y] -= self.learning_rate*derivcosttoweight[i][x][y]
#         for i in range(0,len(biases)):
#             for x in range(0, len(biases[i])):
#                 biases[i][x] -= self.learning_rate*errors[i][x][0]
#         return(weights, biases)
#
#     def output(self,activations):
#         output = ""
#         if activations[self.layers][0] > activations[self.layers][1] and activations[self.layers][0] > activations[self.layers][2]:
#             output = "Iris-setosa"
#         elif activations[self.layers][1] > activations[self.layers][0] and activations[self.layers][1] > activations[self.layers][2]:
#             output = "Iris-versicolor"
#         else:
#             output = "Iris-virginica"
#         return output
#
#
#     def main(self):
#         count = 1
#         temp_count = 0
#         no_count = 0
#         weights = self.total_weights()
#         biases = self.total_bias()
#         while count <= 70 and count > 0:
#             count = 0
#             for i in range(0,99):
#                 z = self.z(weights,biases)
#                 activations = self.activations(z)
#                 first_layer_activations = self.first_layer_activations()
#                 cost = self.cost(activations)
#                 final_layer_error = self.final_layer_error(z,activations)
#                 all_error = self.errorbackprop(weights,final_layer_error,z)
#                 deriv_cost_to_weight = self.derivcosttoweight(activations,first_layer_activations,all_error)
#                 weights = self.gradient_descent(weights,biases,deriv_cost_to_weight,all_error)[0]
#                 biases = self.gradient_descent(weights,biases,deriv_cost_to_weight,all_error)[1]
#                 if self.output(activations) == self.correct_output:
#                     count+=1
#                 Iris = open("Iris.csv", "r")
#                 Irisreader = csv.reader(Iris)
#                 self.test_data_no = random.randint(1, 150)
#                 test_data = []
#                 for row in Irisreader:  # can make this a lot better by doing a binary search rather than just a linear search
#                     if row[0] == str(self.test_data_no):
#                         test_data = [row[1], row[2], row[3], row[4]]  # this is specific to this database so if changing the code this must be edited.
#                         self.correct_output = str(row[5])
#                 Iris.close()
#                 self.inp_act = test_data
#             temp_count += count
#             no_count += 1
#             if no_count == 30:
#                 print(temp_count//no_count)
#                 temp_count = 0
#                 no_count = 0
#
#
#
#
#
#
#
#
#
#
# start = time.process_time()
# flower = Network(2,5,3,0.05)
# # all_weights = flower.total_weights()
# # print(all_weights)
# # all_biases = flower.total_bias()
# # print(all_biases)
# # all_z = flower.z(all_weights,all_biases)
# # print(all_z)
# # activations = flower.activations(all_z)
# # print(activations)
# # first_layer_activations = flower.first_layer_activations()
# # print(first_layer_activations)
# # cost = flower.cost(activations)
# # print(cost)
# # print(flower.correct_output)
# # final_layer_error = flower.final_layer_error(all_z,activations)
# # print(final_layer_error)
# # layer_1_matrix = flower.dicttomat_weights(all_weights,0) # have to remember that this is individual to each layer and have to change each time
# # print(layer_1_matrix)
# # all_error = flower.errorbackprop(all_weights,final_layer_error, all_z)
# # print(all_error)
# # derivcosttoweight = flower.derivcosttoweight(activations,first_layer_activations,all_error)
# # print(derivcosttoweight)
# # gradient_descent = flower.gradient_descent(all_weights,all_biases,derivcosttoweight,all_error)
# # print(gradient_descent[0])
# # print(gradient_descent[1])
# flower.main()
# print(time.process_time() - start)