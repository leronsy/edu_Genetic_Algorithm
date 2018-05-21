import random
import math
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
        return self

    def cross(self, parent1, parent2):
        for i in range(len(self.chromosome)):
            if random.randint(0, 1):
                self.chromosome.append(parent1.chromosome[i])
            else:
                self.chromosome.append(parent2.chromosome[i])

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

    def __init__(self, size, chromosome_length, estimator, features, target, classifier,
                 individ_set_initial=None):
        self.features = features
        self.target = target
        self.size = size
        self.estimator = estimator
        self.classifier = classifier
        self.chromosome_length = chromosome_length
        if not individ_set_initial:
            self.individ_set_initial = list()
            for i in range(self.size):
                individ = Individ(chromosome_length)
                fitness = self.fitness(individ.used_features())
                self.individ_set_initial.append([i, individ, fitness])
        else:
            self.individ_set_initial=individ_set_initial

    def crossing(self, multiplier=2):
        parent1, parent2 = self.find_best_parents()
        self.individ_set_extended = list()
        for i in range(self.size * multiplier):
            individ = Individ(self.chromosome_length, parent1, parent2)
            fitness = self.fitness(individ.used_features())
            self.individ_set_extended.append([i, individ, fitness])

    def mutation(self, probability):
        self.individ_set_mutated = list()
        for indiv_list in self.individ_set_extended:
            i = indiv_list[0]
            individ = indiv_list[1].mutate(probability=probability)
            fitness = self.fitness(individ.used_features())
            self.individ_set_mutated.append([i, individ, fitness])

    def selection(self, selection_limit=None, mutation_probability=0.2, crossing_multiplier=2,
                  new_size=None):
        if not selection_limit:
            selection_limit = self.size
        if not new_size:
            new_size = self.size
        self.crossing(multiplier=crossing_multiplier)
        self.mutation(probability=mutation_probability)
        individ_set_future = self.sort(self.individ_set_mutated)[:selection_limit]

        return Population(size=new_size, chromosome_length=self.chromosome_length,
                          estimator=self.estimator, features=self.features, target=self.target,
                          individ_set_initial=individ_set_future,classifier=self.classifier)

    def sort(self, population):
        return sorted(population, key=lambda ind: ind[2], reverse=True)

    def fitness(self, features_list):
        if features_list:
            self.classifier.fit(self.features[:,features_list],self.target)
            predicted = self.classifier.predict(self.features[:,features_list])
            estimation = self.estimator(predicted, self.target)
        else:
            estimation=float('Inf')
        return estimation

    def find_best_parents(self):
        parent1_list, parent2_list = self.sort(self.individ_set_initial)[:2]
        parent1 = parent1_list[1]
        parent2 = parent2_list[1]
        return parent1, parent2

    def get_info(self):
        for ind in self.individ_set_initial:
            print(ind[1].chromosome)
        return

    def get_best(self):
        best_individ_list = self.sort(self.individ_set_initial)[0]
        individ = best_individ_list[1]
        fitness = best_individ_list[2]
        return individ, fitness


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
        self.generation_list = list()
        self.generation_list.append(self.next_gen(first=True))

    def evolve(self):
        plato = 0
        while len(self.generation_list) < self.generation_limit:
            prev_best_individ, prev_best_score = self.generation_list[-1].get_best()
            self.generation_list.append(self.next_gen())
            current_best_individ, current_best_score = self.generation_list[-1].get_best()
            print(prev_best_score,current_best_score)
            if prev_best_score - current_best_score < EPS:
                plato += 1
                if plato > 2:
                    return current_best_individ
        return current_best_individ

    def next_gen(self, first=False):
        if first:
            generation = Population(size=self.population_size, chromosome_length=self.dimension, estimator=self.estimator,
                                    features=self.features, target=self.target, classifier=self.classifier)
        else:
            generation = self.generation_list[-1].selection(
            mutation_probability=self.mutation_probability,
            crossing_multiplier=self.crossing_multiplier)
        return generation

    def genetic(self):
        return
