import numpy as np
from mnist import MNIST
from NN import NeuralNetwork


def setup(init_pop):
    mndata = MNIST('./mnist_data')
    mndata.gz = True
    images, labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()
    all_data = list(zip(images, labels))
    valid_data = list(zip(test_images, test_labels))
    '''create a population
    loop:
        eval fitness of nn and creating a new  population by
        -pick a parent based on fitness score ,mapped to prop
        -crossover
        -mutation '''
    fitness = {}
    for i in range(0, init_pop):
        population.append(NeuralNetwork(500, 0.01))
    #indices = list(range(len(all_data)))
    test_indices = list(range(len(valid_data)))
    for net in population:
        # validation_idx=np.random.choice(indices, size=784,replace=False)
        test_idx = np.random.choice(test_indices, size=784, replace=False)
        # net.train(createSub(all_data,validation_idx), createSub(valid_data,test_idx), 10)
        loss = net.checkValidation(create_sub(valid_data, test_idx))
        fitness[loss] = net
    sorted(fitness)

    print(crossover(list(fitness.values())[0].weights['W1'], list(fitness.values())[1].weights['W1']))
    print(fitness.keys())


def crossover(weight1, weight2):
    dict_res = {}
    for key, val in weight1:
        father = weight2[key]
        res = np.zeros((val.shape[0], val.shape[1]))
        prob = np.random.uniform(0, 1)
        for i in range(0, val.shape[0]):
            if prob > np.random.rand():
                res[i] = val[i]
            else:
                res[i] = father[i]
        dict_res[key] = res
    return dict_res


def create_sub(data, indicte):
    res = []
    for num in indicte:
        res.append(data[num])
    return res


def mutate(model, func):
    new_weight = {}
    for key, value in model.weights:
        new_weight[key] = np.apply_along_axis(func, 1, np.array(value))
    model.set_weights(new_weight)


population = []
setup(2)
