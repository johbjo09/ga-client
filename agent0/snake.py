import logging
import util

log = logging.getLogger("client.snake")

from collections import deque
from typing import List
from copy import deepcopy

import random
import numpy as np
from ga import GeneticThing
from mlp import MLP
# from overrides import overrides

from base_snake import BaseSnake, Action
from util import Direction, Map, translate_coordinate

log = logging.getLogger("snake")

NUM_INPUTS = 18

class Cell():
    def __init__(self):

        self.cell_north = None
        self.cell_south = None
        self.cell_west = None
        self.cell_east = None
        self.neighbors = None
        
        self.prev = None
        self.start = None

        self.empty = 1
        self.foods = 0
        self.obstacles = 0
        self.heads = 0
        self.body = 0
        self.tails = 0

        self.dist = 1.0
        
        self.is_endpoint = False

    def reset(self):
        self.prev = None
        self.start = None

        self.dist = 1.0

        # Sums along shortest path to F, L, R
        self.empty = 1
        self.foods = 0
        self.obstacles = 0
        self.heads = 0
        self.body = 0
        self.tails = 0

        self.is_endpoint = False

    def set_food(self):
        self.foods = 1.0
        self.empty = 0

    def set_obstacle(self):
        self.obstacles = 1.0
        self.empty = 0
        self.is_endpoint = True

    def set_body(self):
        self.body = 1.0
        self.empty = 0
        self.is_endpoint = True

    def set_head(self):
        self.heads = 1.0
        self.empty = 0
        self.is_endpoint = True

    def set_tail(self):
        self.tails = 1.0
        self.empty = 0
        self.is_endpoint = True

    def sum_up(self, cell):
        self.empty += cell.empty / cell.dist
        self.foods += cell.foods / cell.dist
        self.obstacles += cell.obstacles / cell.dist
        self.heads += cell.heads / cell.dist
        self.body += cell.body / cell.dist
        self.tails += cell.tails / cell.dist

class Snake(BaseSnake):
    def __init__(self, mlp: MLP):
        super().__init__()
        self.name = "agent0"
        self.mlp = mlp
        self.result = None
        self.uid = 0
        self.duration_bfs = 0
        self.cells = None

    def _set_neighbors(self, cell, x, y, gmap: Map):
        cell.neighbors = []
        
        # South
        if y < gmap.height-1:
            cell.cell_south = self.cells[x + (y+1) * gmap.width]
            cell.neighbors.append(cell.cell_south)

        # North
        if y > 0:
            cell.cell_north = self.cells[x + (y-1) * gmap.width]
            cell.neighbors.append(cell.cell_north)

        # West
        if x > 1:
            cell.cell_west = self.cells[x-1 + y * gmap.width]
            cell.neighbors.append(cell.cell_west)

        # East
        if x < gmap.width-1:
            cell.cell_east = self.cells[x+1 + y * gmap.width]
            cell.neighbors.append(cell.cell_east)

    def _make_cells(self, gmap: Map):
        self.cells = [ None ] * gmap.width * gmap.height

        self.board_size = gmap.width * gmap.height

        for x in range(gmap.width):
            for y in range(gmap.height):
                self.cells[x + y*gmap.width] = Cell()

        for x in range(gmap.width):
            for y in range(gmap.height):
                cell = self.cells[x + y*gmap.width]
                self._set_neighbors(cell, x, y, gmap)

    def _load_map(self, gmap: Map):
        for cell in self.cells:
            cell.reset()

        for snake in gmap.game_map['snakeInfos']:
            positions = snake['positions']
            self.cells[positions[0]].set_head()
            self.cells[positions[-1]].set_tail()

            for position in positions[1:-1]:
                self.cells[position].set_body()

        for position in gmap.game_map['obstaclePositions']:
            self.cells[position].set_obstacle()

        for position in gmap.game_map['foodPositions']:
            self.cells[position].set_food()

    def _compute_sums(self, cell_l, cell_f, cell_r, gmap: Map):
        starts = []

        if cell_l:
            cell_l.start = cell_l
            starts.append(cell_l)
        if cell_f:
            cell_f.start = cell_f
            starts.append(cell_f)
        if cell_r:
            cell_r.start = cell_r
            starts.append(cell_r)
        
        # TODO doesn't really work since first cell will have bias
        frontier = deque(starts)

        while len(frontier):
            cell = frontier.popleft()

            for neighbor in cell.neighbors:
                if neighbor.prev is None:
                    neighbor.prev = cell
                    neighbor.start = cell.start
                    neighbor.dist = cell.dist + 1
                    if not neighbor.is_endpoint:
                        frontier.append(neighbor)

                cell.start.sum_up(neighbor)

#    @overrides
    def get_next_action(self, gmap: Map):
        if self.cells is None:
            self._make_cells(gmap)

        self._load_map(gmap)
        
        myself = gmap.get_snake_by_id(self.snake_id)['positions']

        head = self.cells[myself[0]]
        current_direction = self.get_current_direction()

        cell_l, cell_f, cell_r = [ None, None, None ]
        
        if current_direction == Direction.UP:
            cell_l, cell_f, cell_r = head.cell_west, head.cell_north, head.cell_east
        elif current_direction == Direction.RIGHT:
            cell_l, cell_f, cell_r = head.cell_north, head.cell_east, head.cell_south
        elif current_direction == Direction.LEFT:
            cell_l, cell_f, cell_r = head.cell_south, head.cell_west, head.cell_north
        else:  # DOWN
            cell_l, cell_f, cell_r = head.cell_east, head.cell_south, head.cell_west


        if cell_l:
            cell_l.prev = head
        if cell_f:
            cell_f.prev = head
        if cell_r:
            cell_r.prev = head

        self._compute_sums(cell_l, cell_f, cell_r, gmap)

        input_l = None
        input_f = None
        input_r = None
        
        if cell_l:
            input_l = [ cell_l.empty, cell_l.foods, cell_l.obstacles, cell_l.heads, cell_l.body, cell_l.tails ]
        else:
            input_l = [ 0, 0, 0, 0, 0, 0 ]

        if cell_f:
            input_f = [ cell_f.empty, cell_f.foods, cell_f.obstacles, cell_f.heads, cell_f.body, cell_f.tails ]
        else:
            input_f = [ 0, 0, 0, 0, 0, 0 ]

        if cell_r:
            input_r = [ cell_r.empty, cell_r.foods, cell_r.obstacles, cell_r.heads, cell_r.body, cell_r.tails ]
        else:
            input_r = [ 0, 0, 0, 0, 0, 0 ]

        inputs = [ input_l + input_f + input_r ]

        inputs = np.array(inputs)

        # print(inputs)

        output = self.mlp.recall(inputs)

        action = [Action.LEFT, Action.FRONT, Action.RIGHT][output.argmax()]

        return action        

class GeneticSnake(GeneticThing, Snake):
    def __init__(self, r_mutation=0.4, severity=0.5):
        self.uid = None
        
        mlp = MLP(NUM_INPUTS, activation="sigmoid", output="sigmoid")
        mlp.add_layer(6)
        mlp.add_layer(3)
        Snake.__init__(self, mlp)

        self._fitness = 0
        self._r_mutation = r_mutation
        self.severity = severity

    def store_snake(self):
        pass

    @property
    def fitness(self):
        return self._fitness

    def set_fitness(self, fitness):
        self._fitness = fitness

    def mutate(self, r_mutate):
        # Add noise to weights. Noise proportional to 0 < r_mutate < 1
        for i in range(len(self.mlp.W)):
            self.mlp.W[i] = (1  -r_mutate) * self.mlp.W[i] + r_mutate * self._get_randomization(self.mlp.W[i])

    def _get_randomization(self, w):
        # return (np.random.normal(0, 2, num_in + 1))
        w_shape = np.shape(w)
        return (np.random.normal(0, 1, (w_shape[0], w_shape[1])) * self.severity)

    def crosswith(self, that):
        offspring = deepcopy(self)
        offspring.uid = None
        for i in range(len(offspring.mlp.W)):
            w_shape = np.shape(offspring.mlp.W[i])
            mutations = int(self._r_mutation * w_shape[0] * w_shape[1])
            for j in range(mutations):
                k = 0 if w_shape[0] == 1 else random.randint(1, w_shape[0] -1)
                l = 0 if w_shape[1] == 1 else random.randint(1, w_shape[1] -1)
                offspring.mlp.W[i][k][l] = that.mlp.W[i][k][l]
        return offspring

    def distanceto(self, that):
        d = 0
        for i in range(len(self.mlp.W)):
            d += np.sum(np.power(self.mlp.W[i] - that.mlp.W[i], 2))
        return d
    
    def add(self, that):
        for i in range(len(self.mlp.W)):
            self.mlp.W[i] += that.mlp.W[i]

    def divide_by(self, divisor):
        for i in range(len(self.mlp.W)):
            self.mlp.W[i] /= divisor

    def _compute_fitness(self, player_ranks):
        is_alive = None
        points = None
        # rank = None

        for player in player_ranks:
            if player['playerName'] == self.name:
                is_alive = player['alive']
                points = player['points']
                # rank = player['rank']

        alive_bonus = 3 if is_alive else 0

        if is_alive:
            log.debug('Snake %s won :)', self.name)
            
        return self.age + points / 10000.0 + alive_bonus

#    @overrides
    def on_game_result(self, player_ranks):
        self._fitness = self._compute_fitness(player_ranks)
        self.result = self.uid, self._fitness, self.watch_link
