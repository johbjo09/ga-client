import platform

# Inbound
GAME_ENDED = "se.cygni.snake.api.event.GameEndedEvent"
TOURNAMENT_ENDED = "se.cygni.snake.api.event.TournamentEndedEvent"
MAP_UPDATE = "se.cygni.snake.api.event.MapUpdateEvent"
SNAKE_DEAD = "se.cygni.snake.api.event.SnakeDeadEvent"
GAME_STARTING = "se.cygni.snake.api.event.GameStartingEvent"
PLAYER_REGISTERED = "se.cygni.snake.api.response.PlayerRegistered"
INVALID_PLAYER_NAME = "se.cygni.snake.api.exception.InvalidPlayerName"
HEART_BEAT_RESPONSE = "se.cygni.snake.api.response.HeartBeatResponse"
GAME_LINK_EVENT = "se.cygni.snake.api.event.GameLinkEvent"
GAME_RESULT_EVENT = "se.cygni.snake.api.event.GameResultEvent"

# Outbound
REGISTER_PLAYER_MESSAGE_TYPE = "se.cygni.snake.api.request.RegisterPlayer"
START_GAME = "se.cygni.snake.api.request.StartGame"
REGISTER_MOVE = "se.cygni.snake.api.request.RegisterMove"
HEART_BEAT_REQUEST = "se.cygni.snake.api.request.HeartBeatRequest"
CLIENT_INFO = "se.cygni.snake.api.request.ClientInfo"


def player_registration(snake_name):
    return {
        'type': REGISTER_PLAYER_MESSAGE_TYPE,
        'playerName': snake_name,
        'gameSettings': {}
    }


def client_info():
    platform_name = platform.system()
    os_name = "Maybe Linux"
    os_version = "0.0.0"

    if platform_name == "linux" or platform_name == "linux2":
        os_name, os_version, _ = platform.linux_distribution()
    elif platform_name == "darwin" or platform_name == "Darwin":
        os_name = "macOS"
        os_version, _, _ = platform.mac_ver()
    elif platform_name == "win32":
        os_name = "Windows"
        os_version, _, _ = platform.win32_ver()

    return {
        'type': CLIENT_INFO,
        'language': 'Python',
        'languageVersion': platform.python_version(),
        'operatingSystem': os_name,
        'operatingSystemVersion': os_version,
        'clientVersion': 1.0
    }


def register_move(next_move, message):
    return {
        'type': REGISTER_MOVE,
        'direction': next_move,
        'gameTick': message['gameTick'],
        'receivingPlayerId': message['receivingPlayerId'],
        'gameId': message['gameId']
    }


def start_game():
    return {
        'type': START_GAME
    }


def heart_beat(player_id):
    return {'type': HEART_BEAT_REQUEST, 'receivingPlayerId': player_id}
