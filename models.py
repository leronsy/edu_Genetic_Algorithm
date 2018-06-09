import random
import math
import numpy as np
import random
import operator
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


class Individ:

    def __init__(self, length, parent1=None, parent2=None, chromosome=None):

        if chromosome:
            self.chromosome = chromosome
        else:
            self.chromosome = list()

            if parent1 and parent2:
                self.cross(parent1, parent2)
            else:
                self.generate(length)

    # TODO: можно добавить список генов для мутации (по умолчанию - None)
    def mutate(self, probability):
        for i, gen in enumerate(self.chromosome):
            if np.random.choice([True, False], p=[probability, 1 - probability]):
                self.chromosome[i] = 1 - gen
            else:
                self.chromosome[i] = gen
        return Individ(length=len(self.chromosome), chromosome=self.chromosome)

    def cross(self, parent1, parent2):
        for i in range(len(parent1.chromosome)):
            if random.randint(0, 1):
                self.chromosome.append(parent1.chromosome[i])
            else:
                self.chromosome.append(parent2.chromosome[i])

    def generate(self, length):
        self.chromosome = [random.randint(0, 1) for _ in range(length)]

    def used_genes(self):
        genes = list()
        for i, f in enumerate(self.chromosome):
            if f:
                genes.append(i)
        return genes

    def __str__(self):
        # print('[', end=' ')
        # for gen in self.chromosome:
        #     print(gen, end=' ')
        # print(']\n')
        return str(self.chromosome)


class Population:

    def __init__(self, size, chromosome_length, estimator, objects, target, classifier,
                 individ_set_initial=None):
        self.objects = objects
        self.target = target

        self.estimator = estimator
        self.classifier = classifier

        self.size = size
        self.chromosome_length = chromosome_length
        if not individ_set_initial:
            self.individ_set_initial = list()
            for i in range(self.size):
                individ = Individ(chromosome_length)
                fitness = self.fitness(individ.used_genes())
                self.individ_set_initial.append([i, individ, fitness])
        else:
            self.individ_set_initial = individ_set_initial
        self.individ_set_spawned = list()
        self.individ_set_mutated = list()

    def crossing(self, multiplier=2):
        parent1, parent2 = self.find_best_parents()
        for i in range(self.size * multiplier):
            individ = Individ(self.chromosome_length, parent1, parent2)
            fitness = self.fitness(individ.used_genes())
            self.individ_set_spawned.append([i, individ, fitness])

    def mutation(self, probability):
        for indiv_list in self.individ_set_spawned:
            i = indiv_list[0]
            individ = indiv_list[1].mutate(probability=probability)
            fitness = self.fitness(individ.used_genes())
            self.individ_set_mutated.append([i, individ, fitness])

    def selection(self, selection_limit=None, mutation_probability=0.25, crossing_multiplier=2,
                  new_size=None):
        if not selection_limit:
            selection_limit = self.size
        if not new_size:
            new_size = self.size
        self.crossing(multiplier=crossing_multiplier)
        self.mutation(probability=mutation_probability)
        individ_set_future = self.sort(self.individ_set_mutated)[:selection_limit]
        for num, ind in enumerate(individ_set_future):
            individ_set_future[num][0] = num

        return Population(size=new_size, chromosome_length=self.chromosome_length,
                          estimator=self.estimator, objects=self.objects, target=self.target,
                          individ_set_initial=individ_set_future, classifier=self.classifier)

    # TODO: добавить второй параметр сортировки - количество активных генов itertools?
    def sort(self, population):
        return sorted(population, key=lambda ind: ind[2])

    def fitness(self, used_genes):
        if used_genes:
            self.classifier.fit(self.objects[:, used_genes], self.target)
            predicted = self.classifier.predict(self.objects[:, used_genes])
            estimation = self.estimator(predicted, self.target)
        else:
            estimation = float('Inf')
        return estimation

    def find_best_parents(self):
        parent1_list, parent2_list = self.sort(self.individ_set_initial)[:2]
        parent1 = parent1_list[1]
        parent2 = parent2_list[1]
        return parent1, parent2

    # def get_info(self):
    #     for ind in self.individ_set_initial:
    #         ind[1].print_individ()
    #     return

    def get_best(self):
        best_individ_list = self.sort(self.individ_set_initial)[0]
        individ = best_individ_list[1]
        fitness = best_individ_list[2]
        return individ, fitness


class World:

    def __init__(self, objects, target, classifier=LinearRegression(),
                 estimator=mean_squared_error, generation_limit=20,
                 population_size=30, mutation_probability=0.2, crossing_multiplier=2):
        self.train_objects = objects
        self.train_target = target
        self.classifier = classifier
        self.estimator = estimator
        self.generation_limit = generation_limit
        self.population_size = population_size
        self.crossing_multiplier = crossing_multiplier
        self.mutation_probability = mutation_probability
        self.dimension = self.train_objects.shape[1]
        self.generation_list = list()
        self.generation_list.append(self.next_gen(first=True))

    def evolve(self, eps=10 ** -3):
        plato = 0
        while len(self.generation_list) < self.generation_limit:
            last_generation = self.generation_list[-1]
            prev_best_individ, prev_best_score = last_generation.get_best()
            self.generation_list.append(self.next_gen())
            last_generation = self.generation_list[-1]
            cur_best_individ, cur_best_score = last_generation.get_best()
            print(cur_best_individ, cur_best_score)
            if math.fabs(prev_best_score - cur_best_score) < eps:
                plato += 1
                if plato > 2:
                    break

        # print(cur_best_individ.print_individ())
        return cur_best_individ

    def next_gen(self, first=False):
        if first:
            generation = Population(size=self.population_size, chromosome_length=self.dimension,
                                    estimator=self.estimator,
                                    objects=self.train_objects, target=self.train_target,
                                    classifier=self.classifier)
        else:
            last_generation = self.generation_list[-1]
            generation = last_generation.selection(mutation_probability=self.mutation_probability,
                                                   crossing_multiplier=self.crossing_multiplier)
        return generation
