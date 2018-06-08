import numpy as np

sigmoid = lambda x: 1 / (1 + np.exp(-x))

sigmoidPrime = lambda z: np.multiply(np.array(sigmoid(np.array(z))), np.array((1 - sigmoid(np.array(z)))))


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


difftanh = lambda x: 4 / (np.power(np.exp(-x) + np.exp(x), 2))


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def checkValidation(weights,validation_set,ac_fun):
    right_exmp = 0
    for x, y in validation_set:
        val_func = fprop(x, y, weights, ac_fun)
        if ((np.argmax(val_func['h2'])) == int(y)):
            right_exmp = right_exmp + 1
    return right_exmp


def fprop(x, y, params, activation_fun):
    # Follows procedure given in notes
    W1, b1, W2, b2 = [params[key] for key in ('W1', 'b1', 'W2', 'b2')]
    x = np.transpose(np.matrix(x))
    z1 = np.add(np.dot(W1, x), b1)
    h1 = activation_fun(z1)  # activation function
    z2 = np.add(np.dot(W2, h1), b2)
    h2 = softmax(z2)
    loss = -(np.log(h2[int(y)]))
    y_vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    y_vec[int(y)] = 1
    y_vec = np.transpose(np.matrix(y_vec))
    ret = {'x': x, 'y': y_vec, 'z1': z1, 'h1': h1, 'z2': z2, 'h2': h2, 'loss': loss}
    for key in params:
        ret[key] = params[key]
    return ret


def bprop(fprop_cache, diFunc):
    w2 = fprop_cache['W2']
    x = fprop_cache['x']
    y = fprop_cache['y']
    z1 = fprop_cache['z1']
    h1 = fprop_cache['h1']
    h2 = fprop_cache['h2']
    dz2 = np.subtract(h2, y)
    dW2 = np.dot(dz2, np.transpose(h1))  # dL/dz2 * dz2/dw2
    db2 = dz2  # dL/dz2 * dz2/db2
    dz1 = np.multiply(np.array(np.dot(np.transpose(w2), dz2)), np.array(diFunc(z1)))  # dL/dz2 * dz2/dh1 * dh1/dz1
    dW1 = np.dot(dz1, np.transpose(x))  # dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/dw1
    db1 = dz1  # dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/db1
    return {'b1': db1, 'W1': dW1, 'b2': db2, 'W2': dW2}

def create(epochs,layers,lr,train_x,train_y):
    input_size=layers[0]
    prv=input_size
    i=1
    weights = {}
    for hidden_layer in layers[1:]:
        glorot_init = np.sqrt(6 / (1.0 * ( hidden_layer+ prv)))
        weights['W'+str(i)]=np.matrix(np.random.uniform(-1 * glorot_init, glorot_init, (hidden_layer, prv)))
        weights['b'+str(i)]= np.transpose(np.matrix(np.random.uniform(-1 * glorot_init, glorot_init, hidden_layer)))
        i+=1
        prv=hidden_layer
    output_layer=10
    glorot_init = np.sqrt(6 / (1.0 * (output_layer + prv)))
    weights['W'+str(i)]=np.matrix(np.random.uniform(-1 * glorot_init, glorot_init, (output_layer,prv)))
    weights['b'+str(i)]=np.transpose(np.matrix(np.random.uniform(-1 * glorot_init, glorot_init, output_layer)))
    pair_data = list(zip(train_x, train_y))
    np.random.shuffle(pair_data)
    training_size = int(len(pair_data) * 0.8)
    training_set = pair_data[:training_size]
    validation_set = pair_data[training_size:]
    print(lr, epochs)
    for i in range(0, epochs):
        np.random.shuffle(training_set)
        for x, y in training_set:
            functions = fprop(x, y, weights, sigmoid) #forward
            bp_gradients = bprop(functions, sigmoidPrime) #backpropagation
            #update weights
            weights={'W1':(weights['W1'] - lr * bp_gradients['W1']),
                     'b1': (weights['b1'] - lr * bp_gradients['b1'])
                    ,'W2': ( weights['W2'] - lr * bp_gradients['W2'])
                    ,'b2' : (weights['b2'] - lr * bp_gradients['b2'])}
        right=checkValidation(weights,validation_set,sigmoid)
        print(i, "success percentage:" + str((right / float(len(validation_set)) * 100)) + '%')
