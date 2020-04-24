import numpy as np 
import tensorflow as tf
import random

def convert(x):
    finArr = []
    for line in x:
        for px in line:
            finArr.append(px/255.0)
    return np.array([finArr])

class neuralNetwork:
    def __init__(self,layers=[]):
        self.lowerBound = -50
        self.upperBound = 50
        self.L = len(layers)
        self.sizes = layers
        self.weights = list()
        self.biases = list()
        
        for i in range(0,self.L-1):
            self.weights.append(np.random.rand(layers[i+1],layers[i]))
            self.biases.append(np.random.rand(layers[i+1],1))

    def activation(self,x):
        x = np.clip(x, self.lowerBound, self.upperBound)
        z = 1/(1+np.exp(-x))
        return z

    def activation_deriv(self,x):
        x = np.clip(x, self.lowerBound, self.upperBound)
        x = self.activation(x)
        z = x * (1-x)
        return z

    def feedForward(self,a):
        a = np.asarray(a)
        a = convert(a)
        a = a.T
        for i in range(0, self.L-1):
            a = np.dot(self.weights[i],a) + self.biases[i]

        return a

    def backprop(self,x,y):
        x = np.asarray(x)
        x = convert(x)
        x = x.T
        
        des_out = np.zeros((self.sizes[-1],1))
        des_out[y] = 1
        
        modif_w = [np.zeros(np.shape(k)) for k in self.weights]
        modif_b = [np.zeros(np.shape(k)) for k in self.biases]
        #Feed forward
        activ = x
        activations = [x]
        zs = []
        for i,w in enumerate(self.weights):
            z = np.dot(w,activ) + self.biases[i]
            zs.append(z)
            activ = self.activation(z)
            activations.append(activ)

        #Backpropagation + calculate gradient
        delta = (activations[-1] - des_out) * self.activation_deriv(zs[-1])
        modif_b[-1] = delta
        modif_w[-1] = np.dot(modif_b[-1] , activations[-2].T)
        for i in range(2,self.L):
            delta = np.dot(self.weights[-i+1].T,delta) * self.activation(zs[-i])
            modif_b[-i] = delta
            modif_w[-i] = np.dot(delta,activations[-i-1].T)
        return (modif_b,modif_w) 

    def evaluate(self,test_data):
        sum = 0
        (x_test,y_test) = test_data
        for i,x in enumerate(x_test):
            yHat = self.feedForward(x)
            if np.argmax(yHat) == y_test[i]:
                sum+=1

        return sum

    def update_mini_batch(self, batch, eta):
        delta_w = list(np.zeros(np.shape(k)) for k in self.weights)
        delta_b = list(np.zeros(np.shape(k)) for k in self.biases)
        (x_t,y_t) = batch

        for i,x in enumerate(x_t):
            modif_b, modif_w = self.backprop(x,y_t[i])
            for k in range(len(delta_w)):
                delta_w[k] += modif_w[k]
            
            for k in range(len(delta_w)):
                delta_b[k] += modif_b[k]

        for i in range(len(self.weights)):
            self.weights[i] -= (eta/len(batch[0]))*delta_w[i]
        for i in range(len(self.biases)):
            self.biases[i] -= (eta/len(batch[0]))*delta_b[i]
        

        #a[-1] este layer-ul de output
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data = None):
        if test_data: 
            n_test = len(test_data[0])
        n = len(training_data[0])
        for j in range(epochs):
            #first step is to train the NN
            
            (x_t,y_t) = training_data
            c = list(zip(x_t,y_t))
            random.shuffle(c)
            (x_t,y_t) = zip(*c)

            for o in range(0,n,mini_batch_size):
                mini_batch = ( x_t[o:min(o+mini_batch_size,n)] , y_t[o:min(o+mini_batch_size,n)] )
                self.update_mini_batch(mini_batch,eta)
            
            if test_data:
                print("Epoch {0}: {1}/{2}".format(j,self.evaluate(test_data),n_test))
            else:
                print("Epoch {0} completed".format(j))


if __name__ == "__main__":
    
    training_data, test_data= tf.keras.datasets.mnist.load_data()
    net = neuralNetwork([784,100,10])
    net.SGD(training_data, 30, 10, 0.1, test_data=test_data)
            
                        

    #backprop + mini_batch_processing + overall training with epochs -> consult the book at http://neuralnetworksanddeeplearning.com/chap1.html
