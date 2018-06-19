from mnist import MNIST
from network import Network
import numpy as np

mutate_chance = 0.02


def breed(mother, father):
    children = []
    dict_res1 = {}
    network = Network()
    weight1 = mother.weights
    weight2 = father.weights
    for key, val in weight1.items():
        papa = weight2[key]
        res1 = np.zeros((val.shape[0], val.shape[1]))
        for i in range(0, val.shape[0]):
            if 0.5 > np.random.random():
                res1[i] = val[i]
            else:
                res1[i] = papa[i]
        dict_res1[key] = res1
    network.weights = dict_res1
    # Randomly mutate some of the children.
    if mutate_chance > np.random.random():
        network.mutate(mutate_chance)
    children.append(network)
    return children


def train_networks(index, networks, dataset):
    validation_idx = np.random.choice(index, size=50, replace=False)
    sub_set = np.array(dataset)[validation_idx]
    for network in networks:
        network.train(sub_set)


def evolve(graded):
    # Get the number we want to keep for the next gen.
    retain_length = int(len(graded) * 0.4)

    # The parents are every network we want to keep.
    parents = graded[:retain_length]

    # For those we aren't keeping, randomly keep some anyway.
    for individual in graded[retain_length:]:
        if 0.1 > np.random.random():
            parents.append(individual)

    # Now find out how many spots we have left to fill.
    parents_length = len(parents)
    desired_length = len(graded) - parents_length
    children = []

    # Add children, which are bred from two remaining networks.
    while len(children) < desired_length:

        # Get a random mom and dad.
        male = np.random.randint(0, parents_length - 1)
        female = np.random.randint(0, parents_length - 1)

        # Assuming they aren't the same network...
        if male != female:
            male = parents[male]
            female = parents[female]
            # Breed them.
            babies = breed(male, female)

            # Add the children one at a time.
            for baby in babies:
                # Don't grow larger than desired length.
                if len(children) < desired_length:
                    children.append(baby)

    parents.extend(children)

    return parents


def create_population(count):
    pop = []
    for _ in range(0, count):
        network = Network()
        pop.append(network)
    return pop


def generate(generations, population):
    mndata = MNIST('./mnist_data')
    mndata.gz = True
    train_x, train_y = mndata.load_training()
    train_x = np.array(train_x) / 255.0
    test_x, test_labels = mndata.load_testing()
    test_x = np.array(test_x) / 255.0
    dataset = list(zip(train_x, train_y))
    indices = list(range(len(dataset)))
    valid_data = list(zip(test_x, test_labels))
    networks = create_population(population)
    # Evolve the generation.
    best = Network()
    for i in range(generations):
        print("***Doing generation  {} of {}***".format(i + 1, generations))
        train_networks(indices, networks, dataset)
        graded = [(network.loss, network) for network in networks]
        # Sort on the scores.
        graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=False)]
        best = graded[0]
        if i % 10 == 0:
            print(i, " best loss:", best.loss, "best acc", best.acc)
        print('-' * 80)
        # Evolve, except on the last iteration.
        if i != generations - 1:
            # Do the evolution.
            networks = evolve(graded)

    # Sort our final population.
    networks = sorted(networks, key=lambda x: x.loss)
    best.train(valid_data)
    print("test loss:", best.loss, "test  acc", best.acc)


def main():
    generations = 7000  # Number of times to evole the population.
    population = 100  # Number of networks in each generation.
    print("***Evolving {} generations with population {}***".format(generations, population))
    generate(generations, population)


if __name__ == '__main__':
    main()
