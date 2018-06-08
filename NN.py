import numpy as np


class ActivationFunction:
    def __init__(self,func, dfunc) :
        self.func = func
        self.dfunc = dfunc
  

sigmoid = lambda x: 1 / (1 + np.exp(-x))

sigmoidPrime = lambda z: np.multiply(np.array(sigmoid(np.array(z))), np.array((1 - sigmoid(np.array(z)))))

sigmoidAF = ActivationFunction(sigmoid,sigmoidPrime)

tanh =ActivationFunction(np.tanh,lambda x: 4 / (np.power(np.exp(-x) + np.exp(x), 2)))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class NeuralNetwork :
    def __init__(self,a, b, c=10) :
        if isinstance(a,NeuralNetwork) :
           self.input_layer = a.input_layer
           self.hidden_layer = a.hidden_layer 
           self.output_layer = a.output_layer

           self.weights_ih = a.weights_ih
           self.weights_ho = a.weights_ho

           self.bias_h = a.bias_h
           self.bias_o = a.bias_o
        else :
           self.input_layer = a
           self.hidden_layer = b 
           self.output_nodes = c
           glorot_init = np.sqrt(6 / (1.0 * (self.hidden_layer + self.input_layer)))
           self.weights_ih =np.matrix(np.random.uniform(-1 * glorot_init, glorot_init, (self.hidden_layer, self.input_layer)))
           self.weights_ho = np.matrix(np.random.uniform(-1 * glorot_init, glorot_init, (self.output_layer,self.hidden_layer)))
           self.weights_ih.randomize() 
           self.weights_ho.randomize() 

           self.bias_h =np.transpose(np.matrix(np.random.uniform(-1 * glorot_init, glorot_init, self.hidden_layer)).randomize())
           self.bias_o = np.transpose(np.matrix(np.random.uniform(-1 * glorot_init, glorot_init, self.output_layer)).randomize())
        self.lr=self.setLearningRate()
        self.activation_function=ActivationFunction(sigmoid,sigmoidPrime)

    def setLearningRate(learning_rate=0.1):
        lr = learning_rate

    def setActivationFunction(func = sigmoid) :
        activation_function = func



 