import random
import tensorflow as tf
import scipy as sp
from PIL import Image
import numpy as np 


np.seterr(over="ignore")
def img_to_np(path):
    image = np.asarray(Image.open(path))
    finArr = []
    for row in image:
        for pixel in row:
            finArr.append(pixel[0]/255)
    return finArr

lowerBound = -1000
upperBound = 1000

def stable_softmax(X):
    exps = np.exp(X - np.max(X))
    return exps / np.sum(exps)

def cross_entropy(X,y):
    """
    X is the output from fully connected layer (num_examples x num_classes)
    y is labels (num_examples x 1)
    	Note that y is not one-hot encoded vector. 
    	It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
    """
    m = y.shape[0]
    p = softmax(X)
    # We use multidimensional array indexing to extract 
    # softmax probability of the correct label for each sample.
    # Refer to https://docs.scipy.org/doc/numpy/user/basics.indexing.html#indexing-multi-dimensional-arrays for understanding multidimensional array indexing.
    log_likelihood = -np.log(p[range(m),y])
    loss = np.sum(log_likelihood) / m
    return loss

def delta_cross_entropy(X,y):
    """
    X is the output from fully connected layer (num_examples x num_classes)
    y is labels (num_examples x 1)
    	Note that y is not one-hot encoded vector. 
    	It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
    """
    m = y.shape[0]
    grad = softmax(X)
    grad[range(m),y] -= 1
    grad = grad/m
    return grad

def sigmoid(x):
    x = np.clip(x, lowerBound, upperBound)
    z = 1/(1+np.exp(-x))
    return z

def sigmoid_derivative(x):
    x = np.clip(x,lowerBound,upperBound)
    z = sigmoid(x)*(1-sigmoid(x))
    return z

precision = 10

def convert(x):
    finArr = []
    for line in x:
        for px in line:
            finArr.append(round(px/255,precision))

    return finArr 

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0,l,n):
        yield iterable[ndx:min(ndx+n,l)]

class NeuralNetwork:

    def __init__(self):
        self.w1 = np.random.rand(16,784)
        self.b1 = np.random.rand(16,1)
        self.w2 = np.random.rand(16,16)
        self.b2 = np.random.rand(16,1)
        self.w3 = np.random.rand(10,16)
        self.b3 = np.random.rand(10,1)
        self.weights = []
        self.biases = []

    def feedForward(self, x):
        self.z1 = np.dot(self.w1,self.x) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.w2,self.a1) + self.b2
        self.a2 = sigmoid(self.z2)
        self.z3 = np.dot(self.w3,self.a2) + self.b3
        self.yHat = stable_softmax(self.z3)


    def backprop(self,x,y):
        self.x = np.reshape(x, (784,1))
        self.y = np.zeros((10,1))
        self.y[y]=1
        self.feedForward(x)

        self.cost = self.y - self.yHat
        
        self.er3 = self.cost 
        
        self.er2 = np.dot(self.w3.T,self.er3)*sigmoid_derivative(self.z2)
        
        self.er1 = np.dot(self.w2.T,self.er2)*sigmoid_derivative(self.z1)
        

        self.weights.append([self.er3*self.a2.T, self.er2*self.a1.T, self.er1*self.x.T])
        self.biases.append([self.er3, self.er2, self.er1])

    def train(self, x_train,y_train, eta):
        batch_size = len(x_train)
        for i in range(0,batch_size):
            a = convert(x_train[i])
            a = np.asarray(a)
            self.backprop(a,y_train[i])
        #print(self.weights)
        #print(self.biases)
        
        delta_w1=0
        delta_w2=0
        delta_w3=0
        delta_b1=0
        delta_b2=0
        delta_b3=0
        for nw in self.weights:
            delta_w1 = delta_w1 + (eta/batch_size)*nw[2]
            delta_w2 = delta_w2 + (eta/batch_size)*nw[1]
            delta_w3 = delta_w3 + (eta/batch_size)*nw[0]

        for nw in self.biases:
            delta_b1 = delta_b1 + (eta/batch_size)*nw[2]
            delta_b2 = delta_b2 + (eta/batch_size)*nw[1]
            delta_b3 = delta_b3 + (eta/batch_size)*nw[0]
        
        
        self.w1 = self.w1 + delta_w1
        self.w2 = self.w2 + delta_w2
        self.w3 = self.w3 + delta_w3
        
        self.b1 = self.b1 + delta_b1
        self.b2 = self.b2 + delta_b2
        self.b3 = self.b3 + delta_b3
        '''print(delta_w1)
        print('*')
        print(delta_w2)
        print('*')
        print(delta_w3)
        print('*')'''
        print(np.round(self.yHat, 4)*100)
        print('^')
        print(self.y)
        self.weights.clear()
        self.biases.clear()

(x_train,y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
'''
mapData = list(zip(x_train,y_train))
random.shuffle(mapData)
x_train ,y_train = zip(*mapData)
'''
instance = NeuralNetwork()
b_size = 50
eta = 0.1
a=0
for i in range(0,len(x_train),b_size):
    '''if a==0:
        a = int(input("continue for?"))'''
        
    print('*************')
    print(i)
    print(' ')
    instance.train(x_train[i:min(i+b_size,len(x_train))],y_train[i:min(i+b_size,len(y_train))],eta)
    a-=1


wrongCase = 0
totalCase = len(x_test)
for i in range(0,len(x_test)):
    instance.feedForward(x_test[i])
    print("***** "+ i +" ******")
    print(instance.yHat)
    print("^")
    print(y_test)
    if instance.yHat.__index__(max(instance.yHat)) != y_test[i]:
        wrongCase+=1

print((wrongCase/totalCase)*100)
#for x in batch(train_data, b_size)





