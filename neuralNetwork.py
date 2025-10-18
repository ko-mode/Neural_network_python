import numpy as np

# Building Blocks: Neurons

def sigmoid(x):
    # Our activation function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))

# Blueprint for neurons which make up the neural network
class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        # Weight inputs (w), add bias (b), then use the activation function (f)
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)
    
weights = np.array([0, 1])  # w = [0, 1]
bias = 4                    # b = 4
n = Neuron(weights, bias)

x = np.array([2, 3])        # start off with x = 2, 3
print(n.feedforward(x))     # feed forward into the function

# Combining Neurons into a Neural Network

class OurNeuralNetwork:
    '''
    A neural network with:
        - 2 inputs
        - a hidden layer with 2 neurons (h1, h2)
        - an output layer with 1 neuron (o1)
    Each neuron has the same weights and bias:
        - w = [0, 1]
        - b = 0
    '''

    def __init__(self):
        weights = np.array([0,1])
        bias = 0

        # The Neuron class here is from the previous section
        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)

    def feedforward(self, x):
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)

        # The inputs for o1 are the outputs from h1 and h2
        out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))

        return out_o1
    
network = OurNeuralNetwork()
x = np.array([2, 3])
print(network.feedforward(x))

# Training the Neural Network
# We’ll represent Male with a 00 and Female with a 11, and we’ll also shift the data to make it easier to use.
# We shift by arbitrary amounts (133 and 66), normally would shift by mean.

# To measure how good a network is and how to improve it, we use loss
# We use the mean square error (MSE) loss
# Better predictions = Lower loss
# So we basically find the squared error and then add it and divide by n (average) (standard deviation formula basically)
# Training a network = trying to minimise its loss
