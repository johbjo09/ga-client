from time import time
import numpy as np
import math
import random
from abc import ABCMeta, abstractmethod, abstractproperty
from copy import deepcopy

class GeneticThing():
    __metaclass__ = ABCMeta

    @abstractproperty
    def fitness(self):
        pass

    @abstractmethod
    def mutate(self, r_mutate):
        pass

    @abstractmethod
    def crosswith(self, that_thing):
        pass

    @abstractmethod
    def distanceto(self, that_thing):
        pass

    # These are for computing the mean individual
    @abstractmethod
    def add(self, that_thing):
        pass

    @abstractmethod
    def subtract(self, that_thing):
        pass

    @abstractmethod
    def divide_by(self, divisor):
        pass

class GeneticAlgorithm():
    
    def __init__(self, population_size,
                 r_mutation=0.04,
                 apex_stddev = 1):

        self.generation = 0

        # Steady state population size
        self._population_size = population_size
        self._r_mutation = r_mutation
        self.apex_stddev = apex_stddev

        self.population = []
        self.apexes = []

    @property
    def population_size(self):
        population_size = 5 * self._population_size * np.exp(-0.5 * self.generation) + self._population_size

        return int(population_size)

    def _selection_base(self, population_size):
        selection_base = self._population_size - len(self.apexes)

        if selection_base < 1:
            selection_base = 1

        return selection_base
        
        p_selection = 0.8 - 0.5 * np.exp(-self.generation / 5.0)

        selection_base = int(self.population_size * p_selection)


        print("selection_base: " + str(selection_base))

        return selection_base

    def _mutation_rate(self):
        r_mutation = (1.0 - self._r_mutation) * np.exp(-0.005 * self.generation) + self._r_mutation
        return r_mutation

    def append(self, thing):
        self.population.append(thing)

    def __iter__(self):
        return iter(self.population)

    def evolve(self):
        population_size = self.population_size
        selection_base = self._selection_base(population_size)
        r_mutation = self._mutation_rate()

        selection_size = int(population_size / 2.0)
        apex_maxsize = int(0.2 * selection_base)

        if selection_size < 1:
            selection_size = 1

        if apex_maxsize < 1:
            apex_maxsize = 1

        self.population.sort(key=lambda s: -s.fitness)
        self.population = self.population[0:selection_base]

        self.population.extend(self.apexes)

        population_mean = deepcopy(self.population[0])
        for thing in self.population[1:]:
            population_mean.add(thing)
        population_mean.divide_by(len(self.population))

        fitness = [ thing.fitness for thing in self.population ]
        
        sum_fitness = sum(fitness)
        max_fitness = max(fitness)
        mean_fitness = np.mean(fitness)
        stddev_fitness = np.sqrt(np.var(fitness))
        apex_cutoff = mean_fitness + self.apex_stddev * stddev_fitness

        p_fitness = lambda i: fitness[i]/max_fitness

        # Distance to mean individual is measure of "distance"
        population_mean = deepcopy(self.population[0])
        for thing in self.population[1:]:
            population_mean.add(thing)
        population_mean.divide_by(len(self.population))

        distances = [ thing.distanceto(population_mean) for thing in self.population ]
        max_distance = max(distances)

        p_distance = lambda i: distances[i]/max_distance

        # Rank function
        f_rank = lambda i: p_fitness(i)* 0.7 + 0.3 * p_distance(i)
        if max_distance == 0:
            f_rank = lambda i: p_fitness(i)

        rankings = [ f_rank(i) for i in range(len(self.population)) ]

        i_apex = list(filter(lambda i: fitness[i] > apex_cutoff, range(len(self.population))))
        if len(i_apex) > apex_maxsize:
            i_apex = range(apex_maxsize)

        self.apexes = [ deepcopy(self.population[i]) for i in i_apex ]

        print("Generation: {}, mean(fitness): {:.2f}, stddev(fitness): {:.2f}, r_mutation: {:.2f}".format(self.generation,
                                                                                                          mean_fitness,
                                                                                                          stddev_fitness,
                                                                                                          r_mutation))

        for i in i_apex:
            print(" apex - fitness: {:.2f}, distance: {:.2f}, rank: {:.2f}".format(fitness[i], distances[i], rankings[i]))

        next_generation = []

        trials = 0

        if self.generation < 10:
            i_selections = []
            i_selections += i_apex

            while len(i_selections) < selection_size and (trials < (100 * population_size)):
                trials += 1

                i = random.randint(0, len(self.population)-1)
                if i in i_selections:
                    continue

                p_selection = rankings[i]

                if random.random() < p_selection:
                    i_selections.append(i)

            for i1 in i_selections:
                ancestor1 = self.population[i1]
                fitness1 = p_fitness(i1)
                mutant1 = deepcopy(ancestor1)
                mutant1.mutate(r_mutation * (1 - 0.5*fitness1))
                next_generation.append(ancestor1)
                next_generation.append(mutant1)
        else:
            while len(next_generation) < population_size:
                i1 = random.randint(0, len(self.population)-1)
                i2 = random.randint(0, len(self.population)-1)

                p_selection1 = rankings[i1]
                p_selection2 = rankings[i2]

                if random.random() < p_selection1 and random.random() < p_selection2:
                    ancestor1 = self.population[i1]
                    ancestor2 = self.population[i2]

                    fitness1 = p_fitness(i1)
                    fitness2 = p_fitness(i2)

                    offspring1 = ancestor1.crosswith(ancestor2, fitness2/(fitness1+fitness2))
                    offspring2 = ancestor2.crosswith(ancestor1, fitness1/(fitness1+fitness2))
                    offspring2.mutate(1 - np.sqrt(fitness1 * fitness2))
                    
                    next_generation.append(offspring1)
                    next_generation.append(offspring2)

        self.population = next_generation
        self.generation += 1

        return sum_fitness
