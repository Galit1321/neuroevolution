import numpy as np
from mnist import MNIST
from NN import NeuralNetwork


def setup(init_pop):
    population = []
    mndata = MNIST('./mnist_data')
    mndata.gz = True
    images, labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()
    all_data = list(zip(images, labels))
    valid_data = list(zip(test_images, test_labels))
    fitness = {}
    for j in range(0, init_pop):
        population.append(NeuralNetwork(100, 0.01))
    indices = list(range(len(all_data)))
    for i in range(0, 20):
        # test_indices = list(range(len(valid_data)))
        for net in population:
            validation_idx = np.random.choice(indices, size=784, replace=False)
            # test_idx = np.random.choice(test_indices, size=784, replace=False)
            # net.train(createSub(all_data,validation_idx), createSub(valid_data,test_idx), 10)
            loss = net.checkValidation(create_sub(all_data, validation_idx))
            fitness[loss] = net
        children = []
        chosen = roulette_wheel_selection(fitness, int(init_pop * .5))
        raffle_num = list(range(len(chosen)))
        mutation_rate = np.random.uniform(0, 1)
        while len(children) < init_pop:
            p, m = np.random.choice(raffle_num, size=2, replace=False)
            mom = chosen[p]
            pop = chosen[m]
            mutate(mom, mutation_rate)
            mutate(pop, mutation_rate)
            children=children + crossover(mom.weights, pop.weights)
        population = children
        print(i)
    for net in population:
        net.checkValidation(valid_data)
        print(net.accuracy)


def crossover(weight1, weight2):
    child1 = NeuralNetwork(100)
    child2 = NeuralNetwork(100)
    dict_res1 = {}
    dict_res2 = {}
    for key, val in weight1.items():
        father = weight2[key]
        res1 = np.zeros((val.shape[0], val.shape[1]))
        res2 = np.zeros((val.shape[0], val.shape[1]))
        prob = np.random.uniform(0, 1)
        for i in range(0, val.shape[0]):
            if prob > np.random.rand():
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


def roulette_wheel_selection(f_dict, size_select):
    chosen = []
    fitness_dict = f_dict.copy()
    for inter in range(size_select):
        sum = np.sum(list(fitness_dict.keys()))
        p = np.random.uniform(0, sum)
        part_sum = 0.0
        for key,val in fitness_dict.items():
            if part_sum > p:
                chosen.append(val)
                del fitness_dict[key]
                break
            part_sum += key
    return chosen


def mutate(model, mut_rate):
    new_weight = {}
    for key, value in model.weights.items():
        if mut_rate > np.random.uniform(0, 1):
            noise = np.random.normal(0, 1, (value.shape[0], value.shape[1]))
            new_weight[key] = value + noise
        else:
            new_weight[key] = value
    model.set_weights(new_weight)


setup(100)
