import numpy as np


class ActivationFunction:
    def __init__(self, func, dfunc):
        self.func = func
        self.dfunc = dfunc


sigmoid = lambda x: 1 / (1 + np.exp(-x))

sigmoidAF = ActivationFunction(sigmoid,
                               lambda z: np.multiply(np.array(sigmoid(np.array(z))),
                                                     np.array((1 - sigmoid(np.array(z))))))

tanh = ActivationFunction(np.tanh, lambda x: 4 / (np.power(np.exp(-x) + np.exp(x), 2)))


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class NeuralNetwork():
    def __init__(self, a, b, acfunc=sigmoidAF, lr=0.01):
        if isinstance(a, NeuralNetwork):
            self.input_layer = a.input_layer
            self.hidden_layer = a.hidden_layer
            self.output_layer = 10
            self.weights = a.weights
            self.activation_function = a.activation_function
            self.accuracy = 0.0
        else:
            self.input_layer = a
            self.hidden_layer = b
            self.output_layer = 10
            glorot_init = np.sqrt(6 / (1.0 * (self.hidden_layer + self.input_layer)))
            W1 = np.matrix(
                np.random.uniform(-1 * glorot_init, glorot_init, (self.hidden_layer, self.input_layer)))
            W2 = np.matrix(np.random.uniform(-1 * glorot_init, glorot_init, (self.output_layer, self.hidden_layer)))
            b1 = np.transpose(
                np.matrix(np.random.uniform(-1 * glorot_init, glorot_init, self.hidden_layer)))
            b2 = np.transpose(
                np.matrix(np.random.uniform(-1 * glorot_init, glorot_init, self.output_layer)))
            self.weights = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        self.lr = lr
        self.accuracy = 0.0
        self.activation_function = acfunc

    def updateWeights(self, gradients):
        self.weights = {'W1': (self.weights['W1'] - self.lr * gradients['W1']),
                        'b1': (self.weights['b1'] - self.lr * gradients['b1'])
            , 'W2': (self.weights['W2'] - self.lr * gradients['W2'])
            , 'b2': (self.weights['b2'] - self.lr * gradients['b2'])}

    def set_weights(self, weights):
        self.weights = weights

    def forward(self, x, y):
        # Follows procedure given in notes
        W1, b1, W2, b2 = [self.weights[key] for key in ('W1', 'b1', 'W2', 'b2')]
        x = np.transpose(np.matrix(x))
        z1 = np.add(np.dot(W1, x), b1)
        h1 = self.activation_function.func(z1)  # activation function
        z2 = np.add(np.dot(W2, h1), b2)
        h2 = softmax(z2)
        loss = -(np.log(h2[int(y)]))
        y_vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        y_vec[int(y)] = 1
        y_vec = np.transpose(np.matrix(y_vec))
        ret = {'x': x, 'y': y_vec, 'z1': z1, 'h1': h1, 'z2': z2, 'h2': h2, 'loss': loss}

        return ret

    def checkValidation(self, validation_set):
        right_exmp = 0.0
        for x, y in validation_set:
            val_func = self.forward(x, y)
            if ((np.argmax(val_func['h2'])) == int(y)):
                right_exmp = right_exmp + 1.0
        return right_exmp

    def bprop(self, fprop_cache):
        w2 = self.weights['W2']
        x = fprop_cache['x']
        y = fprop_cache['y']
        z1 = fprop_cache['z1']
        h1 = fprop_cache['h1']
        h2 = fprop_cache['h2']
        dz2 = np.subtract(h2, y)
        dW2 = np.dot(dz2, np.transpose(h1))  # dL/dz2 * dz2/dw2
        db2 = dz2  # dL/dz2 * dz2/db2
        dz1 = np.multiply(np.array(np.dot(np.transpose(w2), dz2)),
                          np.array(self.activation_function.dfunc(z1)))  # dL/dz2 * dz2/dh1 * dh1/dz1
        dW1 = np.dot(dz1, np.transpose(x))  # dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/dw1
        db1 = dz1  # dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/db1
        return {'b1': db1, 'W1': dW1, 'b2': db2, 'W2': dW2}

    def train(self, training_set, validation_set, epochs):
        for i in range(0, epochs):
            np.random.shuffle(training_set)
            for x, y in training_set:
                functions = self.forward(x, y)  # forward
                bp_gradients = self.bprop(functions)  # backpropagation
                # update weights
                self.updateWeights(bp_gradients)
            right = self.checkValidation(validation_set)
            self.acc = right / float(len(validation_set)) * 100.0
            print(i, "success percentage:" + str(self.acc) + '%')
