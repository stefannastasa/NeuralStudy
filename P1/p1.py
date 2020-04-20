import numpy as np
from PIL import Image
import numpy as np 

def img_to_np(path):
    image = np.asarray(Image.open(path))
    finArr = []
    for row in image:
        for pixel in row:
            finArr.append(pixel[0]/254)
    return finArr

def sigmoid(x):
    z = 1/(1+np.exp(-x))
    return z

def sigmoid_derivative(x):
    z = x*(1-x)
    return z

class NeuralNetwork:

    def __init__(self):
        self.weights1 = np.random.rand(784,16)
        self.bias1 = np.random.rand(16)
        self.weights2 = np.random.rand(16,16)
        self.bias2 = np.random.rand(16)
        self.weights3 = np.random.rand(16,10)
        self.bias3 = np.random.rand(10)
        self.deriv_cost = []
    
    #   Trebuie facuta o modificare!
    #   Sa iti ia in feedForward datele de input si output si sa faca feedforward pe ele
    #   Dupa care sa iti calculeze costul acelui set de date si sa il introduca in lista de costuri
    #   Backprop: faci average-ul la toata lista si modifici toate weight-urile si bias-urile cum trebuie 
    #   dupa cureti vectorul de costuri si introduci un nou set de date
    def feedForward(self, inp, oup):
        self.x = np.array(inp)
        self.y = np.zeros(10)
        self.y[oup] = 1

        self.layer1 = sigmoid(np.dot(self.x,self.weights1)+self.bias1)
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2)+self.bias2)
        self.out = sigmoid(np.dot(self.layer2,self.weights3)+self.bias3)

        self.deriv_cost.append(2*(self.y-self.out))

#disclaimer - pos sa nu fie corect modul in care am inceput sa fac backprop
    def backprop(self):
        average = np.zeros(10)
        for i in self.deriv_cost:
            average += i

        average = average / len(self.cost)
        d_weights3 = np.dot(self.layer2,sigmoid())

first_instance = NeuralNetwork()
a = img_to_np("P1/pict1.jpg")
first_instance.feedForward(a, 2)
first_instance.backprop()


