import numpy as np


def relu(x):
    return np.maximum(0, x)


sigmoid = lambda x: 1 / (1 + np.exp(-x))


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class NeuralNetwork():
    def __init__(self, b):
        self.input_layer = 784
        self.hidden_layer = 128
        self.output_layer = 10
        glorot_init = np.sqrt(6 / (1.0 * (self.hidden_layer + self.input_layer)))
        W1 = np.matrix(np.random.uniform(-1 * glorot_init, glorot_init, (self.hidden_layer, self.input_layer)))
        W2 = np.matrix(np.random.uniform(-1 * glorot_init, glorot_init, (64, 128)))
        W3 = np.matrix(np.random.uniform(-1 * glorot_init, glorot_init, (self.output_layer, 64)))
        b1 = np.transpose(np.matrix(np.random.uniform(-1 * glorot_init, glorot_init, self.hidden_layer)))
        b2 = np.transpose(np.matrix(np.random.uniform(-1 * glorot_init, glorot_init, 64)))
        b3 = np.transpose(np.matrix(np.random.uniform(-1 * glorot_init, glorot_init, self.output_layer)))
        self.weights = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'b3': b3, 'W3': W3}
        self.accuracy = 0.0


    def set_weights(self, weights):
        self.weights = weights

    def forward(self, x, y,activation_fun):
        # Follows procedure given in notes
        W1, b1, W2, b2, W3, b3 = [self.weights[key] for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3')]
        x = np.transpose(np.matrix(x))
        z1 = np.add(np.dot(W1, x), b1)
        h1 = activation_fun(z1)  # activation function
        z2 = np.add(np.dot(W2, h1), b2)
        h2 = activation_fun(z2)
        z3 = np.add(np.dot(W3, h2), b3)
        h3 = softmax(z3)
        loss = -(np.log(h3[int(y)]))
        ret = {'h3': h3, 'loss': loss}
        return ret

    def checkValidation(self, validation_set, ac_fun):
        right_exmp = 0
        loss = 0.0
        for x, y in validation_set:
            val_func = self.forward(x, y, ac_fun)
            loss += val_func['loss'].item()
            if (np.argmax(val_func['h3'])) == int(y):
                right_exmp = right_exmp + 1
        self.accuracy = right_exmp / float(len(validation_set)) * 100.0
        return loss

