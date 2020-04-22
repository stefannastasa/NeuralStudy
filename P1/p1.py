import tensorflow as tf
from PIL import Image
import numpy as np 


np.seterr(over="ignore")
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
        self.w1 = np.random.rand(16,784)
        self.b1 = np.random.rand(16,1)
        self.w2 = np.random.rand(16,16)
        self.b2 = np.random.rand(16,1)
        self.w3 = np.random.rand(10,16)
        self.b3 = np.random.rand(10,1)

    def feedForward(self, x, y):
        self.x = np.reshape(x, (784,1))
        self.y = np.zeros((10,1), dtype=float)
        self.y[y]=1
        
        self.z1 = np.dot(self.w1,self.x) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.w2,self.a1) + self.b2
        self.a2 = sigmoid(self.z2)
        self.z3 = np.dot(self.w3,self.a2) + self.b3
        self.yHat = sigmoid(self.z3)
        
        return self.yHat

def convert(x):
    finArr = []
    for line in x:
        for px in line:
            finArr.append(px/255)

    return finArr 
    

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
first_instance = NeuralNetwork()
a = np.asarray(convert(x_train[0]))
first_instance.feedForward(a, y_train[0])
print(first_instance.y - first_instance.yHat)
#print(x_train[0])


