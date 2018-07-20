import numpy as np
from mnist import MNIST
import time
import pickle

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def relu(x):
    return np.maximum(0, x)


def forward(weights, x, y, activation_fun):
    # Follows procedure given in notes
    w1, b1, w2, b2, w3, b3 = [weights[key] for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3')]
    x = np.transpose(np.matrix(x))
    z1 = np.add(np.dot(w1, x), b1)
    h1 = activation_fun(z1)  # activation function
    z2 = np.add(np.dot(w2, h1), b2)
    h2 = activation_fun(z2)
    z3 = np.add(np.dot(w3, h2), b3)
    h3 = softmax(z3)
    loss = -(np.log(h3[int(y)]))
    ret = {'h3': h3, 'loss': loss}
    return ret


def check_validation(weights, validation_set, ac_fun):
    right_exmp = 0
    loss = 0.0
    res = ""
    for x, y in validation_set:
        val_func = forward(weights, x, y, ac_fun)
        loss += val_func['loss'].item()
        res += str(np.argmax(val_func['h3']))
        if (np.argmax(val_func['h3'])) == int(y):
            right_exmp = right_exmp + 1
    accuracy = right_exmp / float(len(validation_set)) * 100.0
    loss /= len(validation_set)
    return loss, accuracy


def check_test(weights, validation_set, ac_fun):
    right_exmp = 0
    loss = 0.0
    res = ""
    for x, y in validation_set:
        val_func = forward(weights, x, y, ac_fun)
        loss += val_func['loss'].item()
        res += str(np.argmax(val_func['h3'])) + '\n'
        if (np.argmax(val_func['h3'])) == int(y):
            right_exmp = right_exmp + 1
    accuracy = right_exmp / float(len(validation_set)) * 100.0
    loss /= len(validation_set)
    return loss, accuracy, res


def create_crom(hidden_layer, input_layer=784, output_layer=10):
    glorot_init = np.sqrt(6 / (1.0 * (hidden_layer + input_layer)))
    W1 = np.matrix(np.random.uniform(-1 * glorot_init, glorot_init, (hidden_layer, input_layer)))
    b1 = np.transpose(np.matrix(np.random.uniform(-1 * glorot_init, glorot_init, hidden_layer)))
    glorot_init = np.sqrt(6 / (1.0 * (64 + 128)))
    W2 = np.matrix(np.random.uniform(-1 * glorot_init, glorot_init, (64, 128)))
    b2 = np.transpose(np.matrix(np.random.uniform(-1 * glorot_init, glorot_init, 64)))
    glorot_init = np.sqrt(6 / (1.0 * (64 + 10)))
    W3 = np.matrix(np.random.uniform(-1 * glorot_init, glorot_init, (output_layer, 64)))
    b3 = np.transpose(np.matrix(np.random.uniform(-1 * glorot_init, glorot_init, output_layer)))
    weights = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'b3': b3, 'W3': W3}
    return weights


def setup(init_pop):
    mndata = MNIST('./mnist_data')
    mndata.gz = True
    gen = 12000
    test_x, test_labels = mndata.load_testing()
    test_x = np.array(test_x) / 255.0
    with open('weight2/weights_save16000'+ '.pkl', 'rb') as f:
        best= pickle.load(f)
    valid_data = list(zip(test_x, test_labels))
    #print(gen," best loss:", best[0], "best acc", best[2])
    loss, acc, pred =check_test(best, valid_data, np.tanh)
    with open("weights/weight.txt", 'w') as f:
        for key, value in best.items():
            f.write(key)
            for elem in value:
                f.write(str(elem)+',\n')
            #f.write('%s:%s\n' % (key, val[-1]))
    print("test loss:", loss, "test  acc", acc)




localtime = time.asctime(time.localtime(time.time()))
print("Local current time :", localtime)
setup(50)
localtime = time.asctime(time.localtime(time.time()))
print("Local current time :", localtime)
