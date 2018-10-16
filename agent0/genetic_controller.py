import logging
import numpy as np
# from overrides import overrides

from simulation_controller import SimulationController
from ga import GeneticAlgorithm
from snake import GeneticSnake

log = logging.getLogger("genetic_controller")

class GeneticController(SimulationController):
    def __init__(self, args):
        super().__init__(args)

        population_size = 20
        
        self.ga = GeneticAlgorithm(population_size)
        self.fitness_map = {}

        # population_size = args.population_size
        # args.r_mutation = 0.4

        for i in range(population_size):
            snake = GeneticSnake(r_mutation = 0.2,
                                 severity   = 0.1)

            # if i % 5 == 0:
            #    p.train(P_test, Z_test)
            self.ga.append(snake)

        self._load_genomes()

    def initial_batch(self):
        uid = 0
        for snake in self.ga.population:
            snake.uid = uid
            uid += 1

        return self.ga.population

    def create_batch_from_results(self, results):            
        # Show the results
        print('[Generation #%d] Results:' % self.ga.generation)
        for uid, fitness, watch_link in results:
            self.fitness_map[uid] = fitness
            print(' - #%3d fitness=%10g  => %s' % (uid, fitness, watch_link))

        for snake in self.ga.population:
            snake.set_fitness(self.fitness_map[snake.uid])

        # Evolve the population and go on with the learning
        sum_fitness = self.ga.evolve()

        self._store_genomes()

        uid = 0
        for snake in self.ga.population:
            snake.uid = uid
            uid += 1

        return self.ga.population, sum_fitness
        
    def _store_genomes(self):
        genomes = []
        for i in self.ga.population:
            weights = []
            for W in i.mlp.W:
                weights.append(W)
            genomes += [ weights ]

        np.save("population.npy", np.array(genomes))

    def _load_genomes(self):

        try:
            genomes = np.load("population.npy")
            i = 0
            for g in genomes:
                snake = self.ga.population[i]
                i += 1
                j = 0
                for W in g:
                    snake.mlp.W[j] = W
                    j += 1
        except IOError as e:
            print(e)
    
