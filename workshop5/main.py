import numpy as np
from random import uniform
import math
from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

class NeuralNetwork:
    def __init__(self, size_input, size_output, n_hidden_neurons, learning_rate):
        np.random.seed(2)
        self.w1 = np.random.uniform(low=-0.5, high=0.5, size=(size_input, n_hidden_neurons,))
        self.w2 = np.random.uniform(low=-0.5, high=0.5, size=(n_hidden_neurons, size_output,))
        self.eta = learning_rate
        print(self.w1)
        print(self.w2)
    
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def feedfordward(self):
        self.layer1 = self.sigmoid(np.dot(self.input, self.w1))
        self.output = self.sigmoid(np.dot(self.layer1, self.w2))
    
    def back(self):
        delta_output = (self.target -self.output) * self.output * (1 - self.output)
        delta_layer = self.layer1 * (1 - self.layer1) * np.dot(delta_output, self.w2.T)
        
        updatew1  = self.eta * np.dot(self.input.T, delta_layer)
        updatew2 = self.eta * np.dot(self.layer1.T, delta_output)
        self.w1 += updatew1
        self.w2 += updatew2
    
    def train(self, X_train, y_train):
        for _ in range(10000):
            for x, y in zip(X_train, y_train):
                self.input = np.asarray([x])
                self.target = np.asarray([y])
                self.feedfordward()
                self.back()

    def predict(self, x):
        self.input = np.asarray([x])
        self.feedfordward()
        index_max = np.argmax(self.output)
        print(f'predice: {index_max}')
    

data = load_iris()
X = preprocessing.normalize(data.data)
y = []

print(y)

for d in data.target:
    dn = [0, 0, 0]
    dn[d] = 1
    y.append(dn)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

NN = NeuralNetwork(4, 3, 3, 0.5)
NN.train(X_train, y_train)

for x, y in zip(X_test, y_test):
    NN.predict(x)
    print(f'target {np.argmax(y)}')

NN.predict(np.asarray([6.0, 3.6, 5.2, 2.4]))
NN.predict(np.asarray([4.2, 3.0, 1.2, 0.3]))
NN.predict(np.asarray([6.2, 2.1, 4.3, 1.2]))