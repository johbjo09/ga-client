import os
from concurrent.futures.process import ProcessPoolExecutor

from client import run_simulation

class SnakeGameExecutor(object):
    def __init__(self, args):
        self.hpv = args.host, args.port, args.venue
        self.executor = ProcessPoolExecutor(max_workers=(os.cpu_count()))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.executor.shutdown(wait=True)
        return False

    def run_batch(self, batch):
        params = [(*self.hpv, snake) for snake in batch]
        return self.executor.map(run_simulation, params)
