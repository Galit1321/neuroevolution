import numpy as np
from mnist import MNIST

sigmoid = lambda x: 1 / (1 + np.exp(-x))

sigmoidPrime = lambda z: np.multiply(np.array(sigmoid(np.array(z))), np.array((1 - sigmoid(np.array(z)))))


def relu(x):
    return np.maximum(0, x)


def reluDif(z_1):
    res = np.heaviside(z_1, 0)
    return np.matrix(res)


difftanh = lambda x: 4 / (np.power(np.exp(-x) + np.exp(x), 2))


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def checkValidation(weights, validation_set, ac_fun):
    right_exmp = 0
    for x, y in validation_set:
        val_func = fprop(x, y, weights, ac_fun)
        if (np.argmax(val_func['h3'])) == int(y):
            right_exmp = right_exmp + 1
    return right_exmp


def fprop(x, y, params, activation_fun):
    # Follows procedure given in notes
    W1, b1, W2, b2, W3, b3 = [params[key] for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3')]
    x = np.transpose(np.matrix(x))
    z1 = np.add(np.dot(W1, x), b1)
    h1 = activation_fun(z1)  # activation function
    z2 = np.add(np.dot(W2, h1), b2)
    h2 = activation_fun(z2)
    z3 = np.add(np.dot(W3, h2), b3)
    h3 = softmax(z3)
    loss = -(np.log(h3[int(y)]))
    y_vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    y_vec[int(y)] = 1
    y_vec = np.transpose(np.matrix(y_vec))
    ret = {'x': x, 'y': y_vec, 'z1': z1, 'h1': h1, 'z2': z2, 'h2': h2, 'z3': z3, 'h3': h3, 'loss': loss}
    return ret


def bprop(fprop_cache, diFunc, weights):
    w2 = weights['W2']
    w3 = weights['W3']
    x = fprop_cache['x']
    y = fprop_cache['y']
    z1 = fprop_cache['z1']
    z2 = fprop_cache['z2']
    h1 = fprop_cache['h1']
    h2 = fprop_cache['h2']
    h3 = fprop_cache['h3']
    dz3 = np.subtract(h3, y)
    dW3 = np.dot(dz3, np.transpose(h2))  # dL/dz3 * dz3/dw3
    db3 = dz3  # dL/dz3 * dz3/db3
    dz2 = np.multiply(np.array(np.dot(np.transpose(w3), dz3)), np.array(diFunc(z2)))  # dL/dz3 * dz3/dh2 * dh2/dz2
    # dz2 = np.subtract(h2, y)
    dW2 = np.dot(dz2, np.transpose(h1))  # dL/dz2 * dz2/dw2
    db2 = dz2  # dL/dz2 * dz2/db2
    dz1 = np.multiply(np.array(np.dot(np.transpose(w2), dz2)), np.array(diFunc(z1)))  # dL/dz2 * dz2/dh1 * dh1/dz1
    dW1 = np.dot(dz1, np.transpose(x))  # dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/dw1
    db1 = dz1  # dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/db1
    return {'b1': db1, 'W1': dW1, 'b2': db2, 'W2': dW2, 'b3': db3, 'W3': dW3}


hidden_layer = 128
hidden_layer2 = 64
input_size = 28 * 28
output_layer = 10
lr = 0.005
epochs =50
mndata = MNIST('./mnist_data')
mndata.gz = True
train_x, train_y = mndata.load_training()
train_x = np.array(train_x) / 255.0
test_x, test_labels = mndata.load_testing()
pair_data = list(zip(train_x, train_y))
np.random.shuffle(pair_data)
training_size = int(len(pair_data) * 0.8)
training_set = pair_data[:training_size]
validation_set = pair_data[training_size:]
glorot_init = np.sqrt(6 / (1.0 * (hidden_layer + input_size)))
glorot_init2 = np.sqrt(6 / (1.0 * (hidden_layer + hidden_layer2)))
glorot_init3 = np.sqrt(6 / (1.0 * (hidden_layer + 10)))
W1 = np.matrix(np.random.uniform(-1 * glorot_init, glorot_init, (hidden_layer, input_size)))
W2 = np.matrix(np.random.uniform(-1 * glorot_init2, glorot_init2, (hidden_layer2, hidden_layer)))
W3 = np.matrix(np.random.uniform(-1 * glorot_init3, glorot_init3, (output_layer, hidden_layer2)))
b1 = np.transpose(np.matrix(np.random.uniform(-1 * glorot_init, glorot_init, hidden_layer)))
b2 = np.transpose(np.matrix(np.random.uniform(-1 * glorot_init2, glorot_init2, hidden_layer2)))
b3 = np.transpose(np.matrix(np.random.uniform(-1 * glorot_init3, glorot_init3, output_layer)))
weights = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'b3': b3, 'W3': W3}
print(lr, epochs, hidden_layer, hidden_layer2)
for i in range(0, epochs):
    np.random.shuffle(training_set)
    for x, y in training_set:
        functions = fprop(x, y, weights,relu)  # forward
        bp_gradients = bprop(functions, reluDif, weights)  # backpropagation
        # update weights
        weights = {'W1': (weights['W1'] - lr * bp_gradients['W1']),
                   'b1': (weights['b1'] - lr * bp_gradients['b1']),
                   'W2': (weights['W2'] - lr * bp_gradients['W2']),
                   'b2': (weights['b2'] - lr * bp_gradients['b2']),
                   'W3': (weights['W3'] - lr * bp_gradients['W3']),
                   'b3': (weights['b3'] - lr * bp_gradients['b3'])}
    right = checkValidation(weights, validation_set,relu)
    print(i, "success percentage:" + str((right / float(len(validation_set)) * 100)) + '%')

test_x = np.array(test_x) / 255.0
test_data = list(zip(test_x, test_labels))
right = checkValidation(weights, test_data, relu)
print("test success percentage:" + str((right / float(len(test_data)) * 100)) + '%')
