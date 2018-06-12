import numpy as np
from mnist import MNIST
from NN import NeuralNetwork


def setup(init_pop):
    population = []
    mndata = MNIST('./mnist_data')
    mndata.gz = True
    train_x, train_y = mndata.load_training()
    train_x = np.array(train_x) / 255.0
    test_x, test_labels = mndata.load_testing()
    test_x = np.array(test_x) / 255.0
    all_data = list(zip(train_x, train_y))
    valid_data = list(zip(test_x, test_labels))
    fitness = []
    for j in range(0, init_pop):
        population.append(NeuralNetwork(120, 0.01))
    indices = list(range(len(all_data)))
    for i in range(0, 30):
        print(len(population))
        # test_indices = list(range(len(valid_data)))
        for net in population:
            validation_idx = np.random.choice(indices, size=1000, replace=False)
            # test_idx = np.random.choice(test_indices, size=784, replace=False)
            # net.train(createSub(all_data,validation_idx), createSub(valid_data,test_idx), 10)
            loss = net.checkValidation(create_sub(all_data, validation_idx), np.tanh)
            fitness.append((loss, net))
        population.clear()
        fitness.clear()
        children = []
        chosen = selection(fitness, int(init_pop * .5))
        mutation_rate = 0.1
        for elem in chosen:
            if len(children) == init_pop:
                break
            mom, pop = elem
            mutate(mom, mutation_rate)
            mutate(pop, mutation_rate)
            children = children + crossover(mom.weights, pop.weights)
        chosen.clear()
        population = children
        print(i)
    for net in population:
        net.checkValidation(valid_data, np.tanh)
        print(net.accuracy)


def crossover(weight1, weight2):
    child1 = NeuralNetwork(120)
    child2 = NeuralNetwork(120)
    dict_res1 = {}
    dict_res2 = {}
    for key, val in weight1.items():
        father = weight2[key]
        res1 = np.zeros((val.shape[0], val.shape[1]))
        res2 = np.zeros((val.shape[0], val.shape[1]))
        prob = np.random.random()
        for i in range(0, val.shape[0]):
            if prob > np.random.random():
                res1[i] = val[i]
                res2[i] = father[i]
            else:
                res2[i] = val[i]
                res1[i] = father[i]
        dict_res1[key] = res1
        dict_res2[key] = res2
    child1.set_weights(dict_res1)
    child2.set_weights(dict_res2)
    return [child1, child2]


def create_sub(data, indicte):
    res = []
    for num in indicte:
        res.append(data[num])
    return res


def selection(tuple_lst, desired_length):
    # init "lotto cards"
    chosen = []
    fitness_dict = {}
    i = desired_length
    for elem in tuple_lst:
        key = elem[1]
        fitness_dict[key] = elem[0]
        for j in range(0, i):
            chosen.append(key)
        i -= 1
    if desired_length % 2 != 0:
        desired_length += 1

    # desired_length = int(desired_length / 2)
    # init mates list
    mates = []
    # pair mates to get desired amount of breeds
    while len(mates) < desired_length:
        p, m = np.random.choice(chosen, size=2, replace=False)
        # only create match if both sides are different net
        if p != m:
            mates.append((m, p))
    return mates




def mutate(model, mut_rate):
    new_weight = {}
    for key, value in model.weights.items():
        if mut_rate > np.random.random():
            noise = np.random.normal(scale=0.1, size=(value.shape[0], value.shape[1]))
            new_weight[key] = value + noise
        else:
            new_weight[key] = value
    model.set_weights(new_weight)


setup(50)
