import numpy as np

from NN import NeuralNetwork


def setup(mutation_function):
    for i in range(0,20):
        population.append(NeuralNetwork(28*28,100))

    brain = NeuralNetwork(100, 100)
    child = NeuralNetwork(brain, 100)
    mutate(child, mutation_function)


def mutate(model, func):
    new_weight = {}
    for key, value in model.weights:
        new_weight[key] = np.apply_along_axis(func, 1, np.array(value))
    model.weights = new_weight


population = []