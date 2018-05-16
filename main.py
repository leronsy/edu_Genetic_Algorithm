import random


class Individ:
    def __init__(self, length=0, crossing=False, parent1=None, parent2=None):
        if crossing:
            self.chromosome = self.cross(parent1, parent2)
        else:
            self.chromosome = self.generate_chromosome(length)
        self.vitality_rate = self.fitness()

    def fitness(self):
        return

    def mutate(self):
        return

    def cross(self, parent1, parent2):
        chromosome = list()
        for gen1, gen2 in parent1, parent2:
            if random.randint(0, 1) == 0:
                chromosome.append(gen1)
            else:
                chromosome.append(gen2)
        return chromosome

    def generate_chromosome(self, length):
        chromosome = list()
        for i in range(length):
            chromosome.append(random.randint(0, 1))
        return chromosome

    def get_info(self):
        return


class Population:

    def __init__(self, size):
        """генерирование популяции"""
        self.size = size

    def crossing(self):
        return

    def mutation(self):
        return

    def selection(self):
        return

    def get_info(self):
        return
