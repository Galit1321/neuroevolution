import numpy as np


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def relu(x):
    return np.maximum(0, x)


sigmoid = lambda x: 1 / (1 + np.exp(-x))


class Network():
    def __init__(self, nn_param_choices=None):
        self.acc = 0.0
        self.loss = 0.0
        self.input_layer = 784
        self.hidden_layer1 = 128
        self.hidden_layer2 = 64
        self.output_layer = 10
        glorot_init = np.sqrt(6 / (1.0 * (self.hidden_layer1 + self.input_layer)))
        W1 = np.matrix(np.random.uniform(-1 * glorot_init, glorot_init, (self.hidden_layer1, self.input_layer)))
        W2 = np.matrix(np.random.uniform(-1 * glorot_init, glorot_init, (self.hidden_layer2, self.hidden_layer1)))
        W3 = np.matrix(np.random.uniform(-1 * glorot_init, glorot_init, (self.output_layer, self.hidden_layer2)))
        b1 = np.transpose(np.matrix(np.random.uniform(-1 * glorot_init, glorot_init, self.hidden_layer1)))
        b2 = np.transpose(np.matrix(np.random.uniform(-1 * glorot_init, glorot_init, self.hidden_layer2)))
        b3 = np.transpose(np.matrix(np.random.uniform(-1 * glorot_init, glorot_init, self.output_layer)))
        self.weights = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'b3': b3, 'W3': W3}
        self.af1 = np.random.choice([relu, np.tanh])
        self.af2 = np.random.choice([relu, np.tanh])

    def forward(self, x, y):
        # Follows procedure given in notes
        W1, b1, W2, b2, W3, b3 = [self.weights[key] for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3')]
        x = np.transpose(np.matrix(x))
        z1 = np.add(np.dot(W1, x), b1)
        h1 = self.af1(z1)  # activation function
        z2 = np.add(np.dot(W2, h1), b2)
        h2 = self.af2(z2)
        z3 = np.add(np.dot(W3, h2), b3)
        h3 = softmax(z3)
        loss = -(np.log(h3[int(y)]))
        ret = {'h3': h3, 'loss': loss}
        return ret

    def train(self, dataset):
        right_exmp = 0
        loss = 0.0
        for x, y in dataset:
            val_func = self.forward(x, y)
            loss += val_func['loss'].item()
            if (np.argmax(val_func['h3'])) == int(y):
                right_exmp = right_exmp + 1
        self.acc = right_exmp / float(len(dataset)) * 100.0
        self.loss =loss/ float(len(dataset))

    def mutate(self,mut_rate):
        new_weight = {}
        for key, value in self.weights.items():
            res1 = np.array(value)
            for i in range(0, value.shape[0]):
                if mut_rate > np.random.random():
                    noise = np.random.normal(scale=0.0081, size=value.shape[1])
                    res1[i] += noise
                else:
                    res1[i] = value[i]
            new_weight[key] = np.matrix(res1)
        self.weights=new_weight
