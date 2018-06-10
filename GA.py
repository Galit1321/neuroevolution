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
        population.append(NeuralNetwork(28 * 28, 100))

    for net in population:
        # np.random.choice(all_data, 100, replace=False)
        net.train(all_data, valid_data, 10)
        fitness[net.accuracy] = net
    print(fitness.values())


def mutate(model, func):
    new_weight = {}
    for key, value in model.weights:
        new_weight[key] = np.apply_along_axis(func, 1, np.array(value))
    model.set_weights(new_weight)


population = []
setup(2)
