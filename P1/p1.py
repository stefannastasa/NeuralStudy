import tensorflow as tf
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
        ans = input("Use already trained weights?:")
        if ans.lower() == "no":
            self.weights1 = np.random.rand(784,16)
            self.bias1 = np.random.rand(1,16)
            self.weights2 = np.random.rand(16,16)
            self.bias2 = np.random.rand(1,16)
            self.weights3 = np.random.rand(16,10)
            self.bias3 = np.random.rand(1,10)
            self.cost = {
                "weights": [] ,
                "biases": []  
                }
        else:
            with open("P1/Trained data/weights1", "r") as fil:
                self.weights1 = np.load(fil)
            with open("P1/Trained data/weights2", "r") as fil:
                self.weights2 = np.load(fil)
            with open("P1/Trained data/weights3", "r") as fil:
                self.weights3 = np.load(fil)
            with open("P1/Trained data/bias1", "r") as fil:
                self.bias1 = np.load(fil)
            with open("P1/Trained data/bias2", "r") as fil:
                self.bias2 = np.load(fil)
            with open("P1/Trained data/bias3", "r") as fil:
                self.bias3 = np.load(fil)

    def feedForward(self, inp, oup):
        self.x = np.reshape(inp, (1,784))
        self.y = np.zeros(10)
        self.y[oup] = 1

        self.z1 = np.dot(self.x,self.weights1)+self.bias1
        self.layer1 = sigmoid(self.z1)
        print(self.layer1)
        self.z2 = np.dot(self.layer1,self.weights2)+self.bias2
        self.layer2 = sigmoid(self.z2)
        print(self.layer2)
        self.z3 = np.dot(self.layer2,self.weights3)+self.bias3
        self.output = sigmoid(self.z3)
        print(self.output)  
        #print(self.output)
#disclaimer - pos sa nu fie corect modul in care am inceput sa fac backprop
    def backprop(self):
        d_bias3 =   (-1)*(self.y - self.output) * sigmoid_derivative(self.z3)
        d_weights3 =    np.dot(self.layer2.T,d_bias3)
        #print(np.shape(d_bias3))
        #last layer
        d_bias2 = np.dot(d_bias3,self.weights3.T) * sigmoid_derivative(self.z2)
        d_weights2 = np.dot(self.layer1.T,d_bias2)
        #print(np.shape(d_bias2))

        d_bias1 = np.dot(d_bias2,self.weights2.T) * sigmoid_derivative(self.z1)
        d_weights1 = np.dot(self.x.T,d_bias1)
        
        self.cost["weights"].append([d_weights1,d_weights2,d_weights3])
        self.cost["biases"].append([d_bias1,d_bias2,d_bias3])

    def modif(self):
        w = [0,0,0]
        b = [0,0,0]
        for d_set in self.cost["weights"]:
            for i in range(0,3):
                w[i] += d_set[i]

        for d_set in self.cost["biases"]:
            for i in range(0,3):
                b[i] += d_set[i]

        for i in range(0,3):
            w[i]/=len(self.cost["weights"])
            b[i]/=len(self.cost["biases"])

        self.weights1 += w[0]
        self.weights2 += w[1]
        self.weights3 += w[2]
        '''print(self.weights1)
        print(self.weights2)
        print(self.weights3)
        print('*')
        print(self.bias1)
        print(self.bias2)
        print(self.bias3)'''
        self.bias1 += b[0]
        self.bias2 += b[1]
        self.bias3 += b[2]

def convert(x):
    finArr = []
    for line in x:
        for px in line:
            finArr.append(px/255)

    return finArr 
    

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
first_instance = NeuralNetwork()

print(convert(x_train[0]))

#training part
for i in range(0,100):
    fp = convert(x_train[i])
    a = np.asarray(fp)
    first_instance.feedForward(a,y_train[i])
    first_instance.backprop()
    if i%100 == 0:
        first_instance.modif()


ans = input("do you wish to save the weights and biases?")
if ans.lower =="yes":
    with open("P1/Trained data/weights1", "w") as fil:
        np.save(fil, first_instance.weights1)

    with open("P1/Trained data/weights2", "w") as fil:
        np.save(fil, first_instance.weights2)

    with open("P1/Trained data/weights3", "w") as fil:
        np.save(fil, first_instance.weights3)

    with open("P1/Trained data/bias1", "w") as fil:
        np.save(fil, first_instance.bias1)
    
    with open("P1/Trained data/bias2", "w") as fil:
        np.save(fil, first_instance.bias2)

    with open("P1/Trained data/bias3", "w") as fil:
        np.save(fil, first_instance.bias3)



a = img_to_np("P1/pict2.jpg")
a = np.array(a)
first_instance.feedForward(a,3)
print(first_instance.output)

a = img_to_np("P1/pict1.jpg")
a = np.array(a)
first_instance.feedForward(a,2)
print(first_instance.output)

a = img_to_np("P1/pict3.jpg")
a = np.array(a)
first_instance.feedForward(a,1)
print(first_instance.output)
#print(x_train[0])


