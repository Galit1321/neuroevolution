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
    population = []
    mndata = MNIST('./mnist_data')
    mndata.gz = True
    gen = 25000
    elitism = int(init_pop * .05)
    sel = int(init_pop * .25)
    mutation_rate = 0.05
    train_x, train_y = mndata.load_training()
    train_x = np.array(train_x) / 255.0
    test_x, test_labels = mndata.load_testing()
    test_x = np.array(test_x) / 255.0
    all_data = list(zip(train_x, train_y))
    valid_data = list(zip(test_x, test_labels))
    fitness = []
    print(mutation_rate, elitism, init_pop)
    print(init_pop,gen ,elitism ,sel,mutation_rate)
    for j in range(0, init_pop):
        population.append(create_crom(128))
    indices = list(range(len(all_data)))
    size_sample = 50
    for i in range(0, gen):
        fitness.clear()
        validation_idx = np.random.choice(indices, size=size_sample, replace=False)
        sub_set = np.array(all_data)[validation_idx]
        for crom in population:
            loss, acc = check_validation(crom, sub_set,np.tanh)
            fitness.append((loss, crom, acc))
        fitness = sorted(fitness, key=lambda tup: tup[0])
        best = fitness[0]
        if i % 100 == 0:
            print(i, " best loss:", best[0], "best acc", best[2])
            if i % 1000 == 0:
                with open('weights_save' +str(i)+ '.pkl', 'wb') as f:
                    pickle.dump(best[1], f, pickle.HIGHEST_PROTOCOL)
                size_sample = int(1.2 * size_sample)
        chosen = selection(fitness, sel)
        children = [elem[1] for elem in fitness[:elitism]]
        for elem in chosen:
            if len(children) == init_pop:
                break
            mom, pop = elem
            child1,child2=crossover(mom, pop)
            children = children+[mutate(child1,mutation_rate),mutate(child2,mutation_rate)]
        population = children
    best = fitness[0]
    print(gen," best loss:", best[0], "best acc", best[2])
    loss, acc, pred =(best[1], valid_data, np.tanh)
    with open('weights_save'+ '.pkl', 'wb') as f:
        pickle.dump(best[1], f, pickle.HIGHEST_PROTOCOL)
    with open("weight.txt", 'w') as f:
        for key, value in best[1].items():
            f.write(key)
            for elem in value:
                f.write(str(elem)+',\n')
            #f.write('%s:%s\n' % (key, val[-1]))
    f = open("test_tanh.pred", "w")
    f.write(pred[:-1])
    f.close()
    print("test loss:", loss, "test  acc", acc)


def crossover(weight1, weight2):
    dict_res1 = {}
    dict_res2 = {}
    prob = np.random.random()
    for key, val in weight1.items():
        father = weight2[key]
        res1 = np.zeros((val.shape[0], val.shape[1]))
        res2 = np.zeros((val.shape[0], val.shape[1]))
        for i in range(0, val.shape[0]):
            if prob > np.random.random():
                res1[i] = val[i]
                res2[i] = father[i]
            else:
                res2[i] = val[i]
                res1[i] = father[i]
        dict_res1[key] = res1
        dict_res2[key] = res2
    return [dict_res1, dict_res2]


def selection(tuple_lst, desired_length):
    chosen1 = []
    i = desired_length
    for index in range(0, len(tuple_lst)):
        for j in range(0, i):
            chosen1.append(index)
        i -= 1
    mates = []
    while len(mates) < desired_length:
        p, m = np.random.choice(chosen1, size=2, replace=False)
        if m != p:
            mates.append((tuple_lst[m][1], tuple_lst[p][1]))
    return mates


def mutate(weights, mut_rate):
    new_weight = {}
    for key, value in weights.items():
        res1 = np.array(value)
        for i in range(0, value.shape[0]):
            if mut_rate > np.random.random():
                noise = np.random.normal(scale=0.0081, size=value.shape[1])
                res1[i] += noise
            else:
                res1[i] = value[i]
        new_weight[key] = np.matrix(res1)
    return new_weight


localtime = time.asctime(time.localtime(time.time()))
print("Local current time :", localtime)
print("tanh")
setup(50)
localtime = time.asctime(time.localtime(time.time()))
print("Local current time :", localtime)
