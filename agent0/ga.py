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
    def divide_by(self, divisor):
        pass

class GeneticAlgorithm():
    
    def __init__(self, population_size, p_mutation=0.1, p_descendants=0.6):
        self.p_mutation = p_mutation
        self.p_descendants = p_descendants

        self.population_size = population_size
        self.selection_base = int(population_size * 0.85)
        self.apex = None

        self.population = []
        self.generation = 0

    def append(self, thing):
        self.population.append(thing)

    def __iter__(self):
        return iter(self.population)

    def evolve(self):
        timestamp = time()

        self.population.sort(key=lambda s: -s.fitness)
        self.population = self.population[0:self.selection_base]
        
        fitness = [thing.fitness for thing in self.population]

        print(str(fitness))
        
        sum_fitness = sum(fitness)
        max_fitness = max(fitness)
        mean_fitness = np.mean(fitness)
        stddev_fitness = np.sqrt(np.var(fitness))
        apex_cutoff = mean_fitness + stddev_fitness

        p_fitness = lambda i: fitness[i]/max_fitness
        
        
        # Distance to mean individual is measure of "distance"
        mean = deepcopy(self.population[0])
        for thing in self.population[1:]:
            mean.add(thing)
        mean.divide_by(len(self.population))

        distances = [ thing.distanceto(mean) for thing in self.population ]
        max_distance = max(distances)

        p_distance = lambda i: distances[i]/max_distance

        # Rank function
        f_rank = lambda i: p_fitness(i)* 0.5 + 0.5 * p_distance(i)

        rankings = [ f_rank(i) for i in range(len(self.population)) ]
        
        i_apex = list(filter(lambda i: fitness[i] > apex_cutoff, range(len(self.population))))
        i_selections = []

        l2 = int(0.25 * len(self.population))
        if len(i_apex) > l2:
            i_apex = i_apex[0:l2]

        i_selections += i_apex

        descendants = int(self.p_descendants * len(self.population))
        while len(i_selections) <= descendants:
            i = random.randint(0, len(self.population)-1)
            if i in i_selections:
                continue
            
            # The probability that the individual should be in the next generation
            p_selection = rankings[i]
            
            if np.random.rand() < p_selection:
                i_selections.append(i)

        print("Generation: {}, mean(fitness): {:.2f}, stddev(fitness): {:.2f}".format(self.generation, mean_fitness, stddev_fitness))

        for i in i_apex:
            print(" apex - fitness: {:.2f}, distance: {:.2f}, rank: {:.2f}".format(fitness[i], distances[i], rankings[i]))


        print("Selection: {}".format(time() - timestamp))

        self.apexes = [ self.population[i] for i in i_apex ]

        next_generation = []
        next_generation.extend(self.apexes)
    
        while len(next_generation) < len(self.population):
            i1 = random.choice(i_selections)
            i2 = random.choice(i_selections)
            ancestor1 = deepcopy(self.population[i1])
            ancestor2 = deepcopy(self.population[i2])

            descendant1 = ancestor1.crosswith(ancestor2)

            r_mutation1 = (1 - rankings[i1])
            ancestor1.mutate(r_mutation1)

            r_mutation2 = (1 - rankings[i2])
            ancestor2.mutate(r_mutation2)
            
            descendant2 = ancestor1.crosswith(ancestor2)
            
            next_generation.append(ancestor1)
            next_generation.append(ancestor2)
            next_generation.append(descendant1)
            next_generation.append(descendant2)

        self.population = next_generation
        self.generation += 1

        print("Crossing and mutation: {}".format(time() - timestamp))

        return sum_fitness
