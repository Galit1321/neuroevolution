import numpy as np


class ActivationFunction:
    def __init__(self, func, dfunc):
        self.func = func
        self.dfunc = dfunc


def relu(x):
    res = np.array(x)
    for i in range(0, len(res)):
        res[i] = max(res[i], 0.0)
    return np.matrix(res)


def reluDif(z_1):
    res = np.array(z_1)
    for i in range(0, len(z_1)):
        if res[i] < 0.0:
            res[i] = 0.0
        else:
            res[i] = 1.0
    return np.matrix(res)


sigmoid = lambda x: 1 / (1 + np.exp(-x))

sigmoidAF = ActivationFunction(sigmoid,
                               lambda z: np.multiply(np.array(sigmoid(np.array(z))),
                                                     np.array((1 - sigmoid(np.array(z))))))

tanh = ActivationFunction(np.tanh, lambda x: 4 / (np.power(np.exp(-x) + np.exp(x), 2)))

relu_afunc = ActivationFunction(relu, reluDif)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class NeuralNetwork():
    def __init__(self, b, lr=0.01, acfunc=relu_afunc):
        self.input_layer = 784
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
        loss = 0.0
        for x, y in validation_set:
            val_func = self.forward(x, y)
            loss += val_func['loss'].item()
            if (np.argmax(val_func['h2'])) == int(y):
                right_exmp = right_exmp + 1.0
        loss /= len(validation_set)
        self.accuracy = right_exmp / float(len(validation_set)) * 100.0
        return loss