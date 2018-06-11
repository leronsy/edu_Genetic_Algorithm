import math
from copy import deepcopy

import numpy as np
import random
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import networkx as nx
# import pandas as pd
import matplotlib.pyplot as plt


class Individ:

    def __init__(self, length, id: int, parent1=None, parent2=None, chromosome=None):

        self.chromosome = np.zeros(shape=length).astype(int)
        self.id = id
        #self.
        self.parent1 = parent1
        self.parent2 = parent2
        if chromosome is not None and chromosome.any():
            self.chromosome = chromosome
        elif parent1 and parent2:
            self.cross(parent1, parent2)


        else:
            self.generate(length)

    def generate(self, length):
        self.chromosome = [random.randint(0, 1) for _ in range(length)]

    def cross(self, parent1, parent2):
        for i in range(len(parent1.chromosome)):
            if random.randint(0, 1):
                self.chromosome[i] = parent1.chromosome[i]
            else:
                self.chromosome[i] = parent2.chromosome[i]

    # TODO: можно добавить список генов для мутации (по умолчанию - None)
    def mutate(self, probability):
        for i, gen in enumerate(self.chromosome):
            if np.random.choice([True, False], p=[probability, 1 - probability]):
                self.chromosome[i] = 1 - gen
            else:
                self.chromosome[i] = gen
        return Individ(length=len(self.chromosome), id=self.id, chromosome=self.chromosome,
                       parent1=self.parent1, parent2=self.parent2)

    def used_genes(self):
        genes = list()
        for i, f in enumerate(self.chromosome):
            if f:
                genes.append(i)
        return genes

    def __str__(self):
        res_str = str(self.id) + str(self.chromosome)
        # if self.parent1:
        #     res_str += str(self.parent1.id) + ' ' + str(self.parent2.id)
        return res_str


class Population:

    def __init__(self, size, chromosome_length, estimator, objects, target, test_objects,
                 test_target,
                 classifier, iterator,
                 individ_set_initial=None):
        self.objects = objects
        self.target = target
        self.test_objects = test_objects
        self.test_target = test_target
        self.estimator = estimator
        self.classifier = classifier
        self.id_iter = iterator
        self.size = size
        self.chromosome_length = chromosome_length
        if not individ_set_initial:
            self.individ_set_initial = list()
            for i in range(self.size):
                individ = Individ(chromosome_length, id=next(self.id_iter))
                fitness = self.fitness(individ.used_genes())
                self.individ_set_initial.append([i, individ, fitness])
        else:
            self.individ_set_initial = individ_set_initial
        self.individ_set_spawned = list()
        self.individ_set_mutated = list()

    def crossing(self, multiplier=2):
        parent1, parent2 = self.find_best_parents()
        for i in range(self.size * multiplier):
            individ = Individ(self.chromosome_length, id=next(self.id_iter), parent1=parent1,
                              parent2=parent2)
            fitness = self.fitness(individ.used_genes())
            self.individ_set_spawned.append([i, individ, fitness])

    def mutation(self, probability):
        for indiv_list in self.individ_set_spawned:
            i = indiv_list[0]
            individ = indiv_list[1].mutate(probability=probability)
            fitness = self.fitness(individ.used_genes())
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
        for num, ind in enumerate(individ_set_future):
            individ_set_future[num][0] = num

        return Population(size=new_size, chromosome_length=self.chromosome_length,
                          estimator=self.estimator, objects=self.objects, target=self.target,
                          test_objects=self.test_objects, test_target=self.test_target,
                          individ_set_initial=individ_set_future, classifier=self.classifier,
                          iterator=self.id_iter)

    # TODO: добавить второй параметр сортировки - количество активных генов itertools?
    @staticmethod
    def sort(indiv_set):
        indiv_set = sorted(indiv_set, key=lambda ind: (ind[2], np.count_nonzero(ind[1])))
        prev_i = [0, 1, 2]
        res_set = list()
        for i in indiv_set:
            if not (i[1] == prev_i[1] and i[2] == prev_i[2]):
                res_set.append(i)
        return res_set

    def fitness(self, used_genes):
        if used_genes:
            self.classifier.fit(self.objects[:, used_genes], self.target)
            predicted = self.classifier.predict(self.test_objects[:, used_genes])
            estimation = self.estimator(predicted, self.test_target)
        else:
            estimation = float('Inf')
        return estimation

    def find_best_parents(self):
        parent1_list, parent2_list = self.sort(self.individ_set_initial)[:2]
        parent1 = parent1_list[1]
        parent2 = parent2_list[1]
        return parent1, parent2

    def get_info(self):
        prev_ind = [0, 0, 0]
        for ind in self.individ_set_initial:
            if not (ind[2] == prev_ind[2] and np.array_equal(ind[1].chromosome,
                                                             prev_ind[1].chromosome)):
                print(ind[0], ind[1], ind[2], np.count_nonzero(ind[1].chromosome))
            prev_ind = deepcopy(ind)
        print('\n')

    def get_best(self):
        best_individ_list = self.sort(self.individ_set_initial)[0]
        individ = best_individ_list[1]
        fitness = best_individ_list[2]
        return individ, fitness


class World:

    def __init__(self, objects, test_objects, target, test_target, classifier=LinearRegression(),
                 estimator=mean_squared_error, generation_limit=10,
                 population_size=10, mutation_probability=0.2, crossing_multiplier=2):
        indiv_number = population_size * (crossing_multiplier + 2) * generation_limit
        self.id_iterator = iter(np.arange(1, indiv_number))
        self.objects = objects
        self.target = target
        self.test_objects = test_objects
        self.test_target = test_target
        self.classifier = classifier
        self.estimator = estimator
        self.generation_limit = generation_limit
        self.population_size = population_size
        self.crossing_multiplier = crossing_multiplier
        self.mutation_probability = mutation_probability
        self.dimension = self.objects.shape[1]
        self.generation_list = list()
        self.generation_list.append(self.next_gen(first=True))

    def draw(self):
        g = nx.Graph()
        pos = dict()
        generation_number = len(self.generation_list)
        node_number = generation_number*self.population_size
        individ_number = generation_number*self.population_size*(2+self.crossing_multiplier)
        fitness = np.full(shape=individ_number, fill_value=float('inf'))
        colors = ['green']*node_number
        labels = dict()

        for n, gnr in enumerate(self.generation_list):
            for i, ind in enumerate(gnr.individ_set_initial):
                fitness[ind[1].id] = ind[2]

        min_list = np.where(fitness == fitness.min())
        print(min_list)
        cntr = 0
        for n, gnr in enumerate(self.generation_list):
            for i, ind in enumerate(gnr.individ_set_initial):
                g.add_node(ind[1].id, size=10)
                x = i + 1
                y = n
                pos[ind[1].id] = [x, y]
                fitness[ind[1].id] = ind[2]
                labels[ind[1].id] = str('{0:5.3f}\n{1}'.format(ind[2], ind[1].id))
                #print(type(min_list[0]))
                #print((ind[1].id in min_list.any)==True)
                if ind[1].id in min_list[0]:
                    colors[cntr] = 'blue'
                if ind[1].parent1:
                    g.add_edge(ind[1].id, ind[1].parent1.id)
                    g.add_edge(ind[1].id, ind[1].parent2.id)
                cntr+=1


        #pos = nx.spring_layout(g, iterations=100)
        plt.figure(figsize=(self.population_size,generation_number), dpi=300)
        plt.xticks(range(1,self.population_size+1))
        plt.yticks(range(1, generation_number))

        nx.draw_networkx(g, pos, with_labels=True, node_color=colors, edge_color='grey',font_size=10,font_weight='regular', labels=labels)
        plt.show()

    def evolve(self, eps=10 ** -3):
        plato = 0
        cur_best_individ = None


        while len(self.generation_list) < self.generation_limit:
            last_generation = self.generation_list[-1]
            prev_best_individ, prev_best_score = last_generation.get_best()
            self.generation_list.append(self.next_gen())
            last_generation = self.generation_list[-1]
            cur_best_individ, cur_best_score = last_generation.get_best()
            if math.fabs(prev_best_score - cur_best_score) < eps:
                plato += 1
                if plato > 3:
                    break
            # last_generation.get_info()

            # print('>>',
            #       cur_best_individ, cur_best_score, np.count_nonzero(cur_best_individ.chromosome))

        return cur_best_individ

    def next_gen(self, first=False):
        if first:
            generation = Population(size=self.population_size, chromosome_length=self.dimension,
                                    estimator=self.estimator,
                                    objects=self.objects, target=self.target,
                                    test_objects=self.test_objects, test_target=self.test_target,
                                    classifier=self.classifier, iterator=self.id_iterator)
        else:
            last_generation = self.generation_list[-1]
            generation = last_generation.selection(mutation_probability=self.mutation_probability,
                                                   crossing_multiplier=self.crossing_multiplier)
        return generation
