import logging
from enum import Enum

from util import Map, Direction

log = logging.getLogger("base_snake")

class Action(Enum):
    FRONT = 1
    LEFT = 2
    RIGHT = 3


ACTION_TO_MOVE = {
    Direction.UP: {  # Current direction: UP
        Action.FRONT: Direction.UP,
        Action.LEFT: Direction.LEFT,
        Action.RIGHT: Direction.RIGHT,
    },
    Direction.RIGHT: {  # Current direction: RIGHT
        Action.FRONT: Direction.RIGHT,
        Action.LEFT: Direction.UP,
        Action.RIGHT: Direction.DOWN,
    },
    Direction.DOWN: {  # Current direction: DOWN
        Action.FRONT: Direction.DOWN,
        Action.LEFT: Direction.RIGHT,
        Action.RIGHT: Direction.LEFT,
    },
    Direction.LEFT: {  # Current direction: LEFT
        Action.FRONT: Direction.LEFT,
        Action.LEFT: Direction.DOWN,
        Action.RIGHT: Direction.UP,
    }
}


class BaseSnake(object):
    def __init__(self):
        self.name = None
        self.snake_id = None
        self.watch_link = None

        self.moves = []
        self.actions = []
        self.age = 0

    def on_player_registered(self, snake_id):
        self.snake_id = snake_id
        # log.debug('Player "%s:%s" registered successfully',
        #           self.name, self.snake_id)

    @staticmethod
    def on_game_starting():
        log.debug('Game is starting!')

    def get_next_action(self, game_map: Map):
        raise NotImplementedError()

    def get_current_direction(self):
        if len(self.moves) > 0:
            return self.moves[-1]
        return Direction.UP

    def get_next_move(self, game_map: Map):
        self.age += 1

        action = self.get_next_action(game_map)
        current_direction = self.get_current_direction()
        move = ACTION_TO_MOVE[current_direction][action]

        self.actions.append(action)
        self.moves.append(move)

        return move

    def on_game_link(self, link):
        self.watch_link = link

    @staticmethod
    def on_snake_dead(reason):
        log.debug('Game ended because: %s', reason)

    @staticmethod
    def on_invalid_player_name():
        log.fatal('Player name is invalid, try another!')

    @staticmethod
    def on_game_result(player_ranks):
        log.info('Game result:')
        for player in player_ranks:
            is_alive = 'alive' if player['alive'] else 'dead'
            log.info('%d. %d pts\t%s\t(%s)' %
                     (player['rank'], player['points'], player['playerName'],
                      is_alive))

    @staticmethod
    def on_game_ended():
        # log.debug('The game has ended!')
        pass
