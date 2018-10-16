import asyncio
import json
import logging

from autobahn.asyncio.websocket import (WebSocketClientFactory,
                                        WebSocketClientProtocol)

import messages, util

log = logging.getLogger("client")

loop = asyncio.get_event_loop()


class SnakebotProtocol(WebSocketClientProtocol):
    snake = None

    def __init__(self):
        super().__init__()
        self.__done = loop.create_future()

        self.routing = {
            messages.GAME_ENDED: self._game_ended,
            messages.TOURNAMENT_ENDED: self._tournament_ended,
            messages.MAP_UPDATE: self._map_update,
            messages.SNAKE_DEAD: self._snake_dead,
            messages.GAME_STARTING: self._game_starting,
            messages.PLAYER_REGISTERED: self._player_registered,
            messages.INVALID_PLAYER_NAME: self._invalid_player_name,
            messages.HEART_BEAT_RESPONSE: self._heart_beat_response,
            messages.GAME_LINK_EVENT: self._game_link,
            messages.GAME_RESULT_EVENT: self._game_result
        }

    def onOpen(self):
        # log.info("connection is open")
        self._send(messages.client_info())
        self._send(messages.player_registration(self.snake.name))

    def onMessage(self, payload, is_binary):
        assert not is_binary
        if is_binary:
            log.error('Received binary message, ignoring...')
            return

        msg = json.loads(payload.decode())
        # log.debug("Message received: %s", msg)

        self._route_message(msg)

    def onClose(self, was_clean, code, reason):
        # log.info("Socket is closed!")
        if reason:
            log.error(reason)

    def _send(self, msg):
        # log.debug("Sending message: %s", msg)
        self.sendMessage(json.dumps(msg).encode(), False)

    def _route_message(self, msg):
        fun = self.routing.get(msg['type'], None)
        if fun:
            fun(msg)
        else:
            self._unrecognized_message(msg)

    def _game_ended(self, msg):
        self.snake.on_game_ended()

        # log.debug('Sending close message to websocket')
        self.sendClose(code=self.CLOSE_STATUS_CODE_NORMAL)

        # log.info("WATCH AT: %s", self.watch_game_at)

    def _tournament_ended(self, msg):
        raise RuntimeError('Tournaments not supported!')

    def _map_update(self, msg):
        direction = self.snake.get_next_move(util.Map(msg['map']))
        self._send(messages.register_move(str(direction), msg))

    def _snake_dead(self, msg):
        self.snake.on_snake_dead(msg['deathReason'])

    def _game_starting(self, msg):
        self.snake.on_game_starting()

    def _player_registered(self, msg):
        self._send(messages.start_game())

        self.is_tournament = msg['gameMode'].upper() == 'TOURNAMENT'
        if self.is_tournament:
            raise RuntimeError('Tournaments not supported!')

        player_id = msg['receivingPlayerId']
        self.snake.on_player_registered(player_id)

    def _invalid_player_name(self, msg):
        self.snake.on_invalid_player_name()

    def _heart_beat_response(self, msg):
        pass

    def _game_link(self, msg):
        link = msg["url"]
        # log.info('Watch game at: %s', link)
        self.watch_game_at = link
        self.snake.on_game_link(link)

    def _game_result(self, msg):
        self.snake.on_game_result(msg['playerRanks'])

    @staticmethod
    def _unrecognized_message(msg):
        log.error('Received unrecognized message: %s', msg)

    async def _send_heart_beat(self, player_id):
        while True:
            self._send(messages.heart_beat(player_id))
            await asyncio.sleep(2)

    def connection_lost(self, exc):
        self.__done.set_result(None)

    def wait_connection_lost(self):
        return self.__done


def run_simulation(args):
    host, port, venue, snake = args

    factory = WebSocketClientFactory(u"ws://%s:%s/%s" % (host, port, venue))
    factory.protocol = SnakebotProtocol

    SnakebotProtocol.snake = snake

    # https://stackoverflow.com/a/51089229/2209243
    connection_coro = loop.create_connection(factory, host, port)
    transport, protocol = loop.run_until_complete(connection_coro)
    loop.run_until_complete(protocol.wait_connection_lost())

    protocol.dropConnection(abort=True)
    transport.close()

    return snake.result
