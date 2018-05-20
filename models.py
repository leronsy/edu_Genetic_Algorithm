import random
# import math
import numpy as np
import random
import operator
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

EPS = 10 ** (-1)


class Individ:

    def __init__(self, length=0, parent1=None, parent2=None):

        self.chromosome = list()

        if parent1 and parent2:
            self.cross(parent1, parent2)
        else:
            self.generate_chromosome(length)

    def mutate(self, probability):
        for i, gen in enumerate(self.chromosome):
            if np.random.choice([True, False], p=[probability, 1 - probability]):
                self.chromosome[i] = 1 - gen
            else:
                self.chromosome[i] = gen

    def cross(self, parent1, parent2):
        for gen1, gen2 in parent1, parent2:
            if random.randint(0, 1):
                self.chromosome.append(gen1)
            else:
                self.chromosome.append(gen2)

    def generate_chromosome(self, length):
        self.chromosome = [random.randint(0, 1) for _ in range(length)]

    def info(self):
        print(self.chromosome, '\n')

    def used_features(self):
        used_features = list()
        for i, f in enumerate(self.chromosome):
            if f:
                used_features.append(i)
        return used_features


class Population:

    def __init__(self, size, chromosome_length, estimator, features, target,
                 individ_set_initial=None):
        self.features = features
        self.target = target
        self.size = size
        self.estimator = estimator

        self.chromosome_length = chromosome_length
        if not individ_set_initial:
            self.individ_set_initial = dict()
            for key in range(self.size):
                individ = Individ(chromosome_length)
                fitness = self.fitness(individ.used_features())
                self.individ_set_initial[key] = [individ, fitness]

    def crossing(self, multiplier=2):
        parent1, parent2 = self.find_best_parents()
        self.individ_set_extended = dict()
        for key in range(self.size * multiplier):
            individ = Individ(self.chromosome_length, parent1, parent2)
            fitness = self.fitness(individ.used_features())
            self.individ_set_extended[key] = [individ, fitness]

    def mutation(self, probability):
        self.individ_set_mutated = dict()
        for key, value in self.individ_set_extended.items():
            individ = value[0].mutate(probability=probability)
            fitness = self.fitness(individ.used_features())
            self.individ_set_mutated[key] = [individ, fitness]

    def selection(self, selection_limit=None, mutation_probability=0.2, crossing_multiplier=2,new_size=None):
        if not selection_limit:
            selection_limit = self.size
        if not new_size:
            new_size = self.size
        self.crossing(multiplier=crossing_multiplier)
        self.mutation(probability=mutation_probability)
        self.individ_set_mutated = self.sort(self.individ_set_mutated)

        individ_set_future = {key:self.individ_set_mutated[key] for key in range(new_size) if key in self.individ_set_mutated}
        return Population(size=new_size, chromosome_length=self.chromosome_length,
                          estimator=self.estimator, features=self.features, target=self.target,
                          individ_set_initial=individ_set_future)

    def sort(self, population=None,cut=None):
        return sorted(self.individ_set_mutated.keys(),
                                          key=lambda x: self.individ_set_mutated[x][1],
                                          reverse=True)

    def fitness(self, features_list):
        return self.estimator(self.features[:, features_list], self.target)

    def find_best_parents(self):
        search_field=self.individ_set_initial
        self.sort(search_field)
        i1 = mse_sorted[0][0]
        i2 = mse_sorted[1][0]
        return self.individ_list[i1], self.individ_list[i2]

    def get_info(self):
        return


class World:

    def __init__(self, features, target, classifier=LinearRegression(),
                 estimator=mean_squared_error, generation_limit=10,
                 population_size=20, mutation_probability=0.25, crossing_multiplier=2):
        self.features = features
        self.target = target
        self.classifier = classifier
        self.estimator = estimator
        self.generation_limit = generation_limit
        self.population_size = population_size
        self.crossing_multiplier = crossing_multiplier
        self.mutation_probability = mutation_probability
        self.dimension = self.features.shape[1]
        self.generation_list = list().append(self.next_gen(first=True))
        self.next_gen(first=True)

    def evolve(self):
        return

    def next_gen(self, first=False):
        generation = None
        if first:
            generation = Population(self.population_size, self.dimension, self.estimator,
                                    features=self.features, target=self.target)
        else:
            generation = self.generation_list[-1].selection(
                mutation_probability=self.mutation_probability,
                crossing_multiplier=self.crossing_multiplier)

    def genetic(self):
        return
