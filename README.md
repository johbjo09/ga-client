# Pythonidae
> The Pythonidae, commonly known simply as pythons, from the Greek word python
> (πυθων), are a family of nonvenomous snakes found in Africa, Asia, and
> Australia. Among its members are some of the largest snakes in the world.
> Eight genera and 31 species are currently recognized.
> -- Wikipedia


## Running the client
You need to have [`pipenv`](pipenv) installed as that is how we manage both
dependencies and virtualenvs for the project. To install `pipenv`:
```
sudo pip install pipenv
```

Once you have `piipenv` installed you can use it to download all dependencies
and activate the virtualenv:
```
pipenv shell
```

This will put you in a shell with the python environment activated, allowing you
to run the client:
```
python client/client.py
```

You can also run the tests simply by running:
```
pytest
```

## Modifying the bot
The code for the bot can be found in `client/snake.py` and the behavior of the
bot is controlled by function `get_next_move`, which receives a `game_map` (
of the type `util.Map`) and is expected to respond with a `util.Direction`
object.

If you want to use multiple different snake implementations and switch between
them you can change which Snake class is used by modifying the `get_snake`
function.

## Connecting to a different host or port
You can modify the host, port and log level of the client via flags to
`client.py`, for possible flags and default values see the `--help`:
```
python client/client.py --help
```
