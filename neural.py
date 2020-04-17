import numpy as np
import math

def sigmoid(x):
    z = 1/(1+np.exp(-x))
    return z

def sigmoid_derivative(x):
    z = x*(1-x)
    return z

class NeuralNetwork:
    def __init__(self, x, y):
        self.input          = x
        self.weights1       = np.random.rand(self.input.shape[1],5)
        self.weights2       = np.random.rand(5,1)
        self.y              = y
        self.output         = np.zeros(self.y.shape)


    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))


    def backprop(self):
        matr = 2*(self.y-self.output) * sigmoid_derivative(self.output)
        abc = np.dot(matr, self.weights2.T)
        d_weights2 = np.dot(self.layer1.T, matr)
        d_weights1 = np.dot(self.input.T, abc * sigmoid_derivative(self.layer1))
        self.weights1 += d_weights1
        self.weights2 += d_weights2
    
    def train(self):
        for i in range(1,1500):
            self.feedforward()
            self.backprop()


def main():
    x = np.array(([1,1],[0,0],[1,0],[0,1]))
    y = np.array(([1],[1],[0],[0]))
    first_instance = NeuralNetwork(x,y)
    first_instance.train()



    first_instance.input = np.array([1,1])
    first_instance.feedforward()

    print(first_instance.output)
main()


