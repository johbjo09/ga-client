import math

from enum import Enum

def abs_to_rel_quarters(current_direction, north, east, south, west):
    if current_direction == Direction.UP:
        front, right, back, left = north, east, south, west
    elif current_direction == Direction.RIGHT:
        front, right, back, left = east, south, west, north
    elif current_direction == Direction.LEFT:
        front, right, back, left = west, north, east, south
    else:  # DOWN
        front, right, back, left = south, west, north, east

def translate_position(position, map_width):
    y = math.floor((position / map_width))
    x = math.floor(math.fabs(position - y * map_width))

    return (x, y)


def translate_positions(positions, map_width):
    return [translate_position(pos, map_width) for pos in positions]


def translate_coordinate(coordinate, map_width):
    x, y = coordinate
    return x + y * map_width


def translate_coordinates(coordinates, map_width):
    return [translate_coordinate(c, map_width) for c in coordinates]


def get_manhattan_distance(start, goal):
    x1, y1 = start
    x2, y2 = goal

    x = math.fabs(x1 - x2)
    y = math.fabs(y1 - y2)

    return (x, y)


def get_euclidian_distance(start, goal):
    x1, y1 = start
    x2, y2 = goal

    x = math.pow((x1 - x2), 2)
    y = math.pow((y1 - y2), 2)

    return math.floor(math.sqrt(x + y))


def is_within_square(coord, nw_coord, se_coord):
    x, y = coord
    nw_x, nw_y = nw_coord
    se_x, se_y = se_coord

    return x >= nw_x and x <= se_x and y >= nw_y and y <= se_y


class TileType(Enum):
    EMPTY = ("EMPTY", True)
    WALL = ("WALL", False)
    FOOD = ("FOOD", True)
    OBSTACLE = ("OBSTACLE", False)
    SNAKE_HEAD = ("SNAKE_HEAD", False)
    SNAKE_BODY = ("SNAKE_BODY", False)
    SNAKE_TAIL = ("SNAKE_TAIL", False)

    def __init__(self, id, movable):
        self.id = id
        self.movable = movable

    def __str__(self):
        return self.id


class Tile(object):
    def __init__(self, tile_type, coordinate):
        self.tile_type = tile_type
        self.coordinate = coordinate


class Direction(Enum):
    DOWN = ("DOWN", (0, 1))
    UP = ("UP", (0, -1))
    LEFT = ("LEFT", (-1, 0))
    RIGHT = ("RIGHT", (1, 0))

    def __init__(self, id, movement_delta):
        self.id = id
        self.movement_delta = movement_delta

    def __str__(self):
        return self.id


class Map(object):
    def __init__(self, game_map):
        self.game_map = game_map
        self.width = game_map['width']
        self.height = game_map['height']

    def get_snake_by_id(self, snake_id):
        return next((s for s in self.game_map['snakeInfos']
                     if s['id'] == snake_id), None)

    def get_tile_at(self, coordinate):
        position = translate_coordinate(coordinate, self.width)

        snake_at_pos = None
        for snake in self.game_map['snakeInfos']:
            if position in snake['positions']:
                snake_at_pos = snake

        tile_type = TileType.EMPTY
        if snake_at_pos:
            if position == snake_at_pos['positions'][0]:
                tile_type = TileType.SNAKE_HEAD
            elif position == snake_at_pos['positions'][-1]:
                tile_type = TileType.SNAKE_TAIL
            else:
                tile_type = TileType.SNAKE_BODY
        elif position in self.game_map['obstaclePositions']:
            tile_type = TileType.OBSTACLE
        elif position in self.game_map['foodPositions']:
            tile_type = TileType.FOOD
        elif self.is_coordinate_out_of_bounds(coordinate):
            tile_type = TileType.WALL

        return Tile(tile_type, coordinate)

    def is_tile_available_for_movement(self, coordinate):
        tile = self.get_tile_at(coordinate)

        return (tile.tile_type == TileType.EMPTY or
                tile.tile_type == TileType.FOOD)

    def can_snake_move_in_direction(self, snake_id, direction):
        snake = self.get_snake_by_id(snake_id)
        x, y = translate_position(snake['positions'][0], self.width)
        xd, yd = direction.movement_delta

        return self.is_tile_available_for_movement((x + xd, y + yd))

    def is_coordinate_out_of_bounds(self, coordinate):
        x, y = coordinate
        return x < 0 or x >= self.width or y < 0 or y >= self.height
