import argparse
import logging

import colorlog

from genetic_controller import GeneticController

log_levels = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'fatal': logging.FATAL
}
log_names = list(log_levels)


def set_up_logging(args):
    handler = colorlog.StreamHandler()
    formatter = colorlog.ColoredFormatter(
        fmt=('%(log_color)s[%(asctime)s %(levelname)8s] --'
             ' %(message)s (%(filename)s:%(lineno)s)'),
        datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)

    if args.log_level in log_levels:
        logging.basicConfig(
            handlers=[handler],
            level=log_levels[args.log_level]
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Python client for Cygni's snakebot competition",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Server flags
    parser.add_argument(
        '-r',
        '--host',
        default='localhost',
        help='The host to connect to')
    parser.add_argument(
        '-p', '--port', default='8080', help='The port to connect to')
    parser.add_argument(
        '-v',
        '--venue',
        default='training',
        choices=['training', 'tournament'],
        help='The venue (training or tournament)')

    # Verbosity
    parser.add_argument(
        '-l',
        '--log-level',
        default=logging.INFO,
        choices=log_names,
        help='The log level for the client')

    # Genetic algorithm parameters
    parser.add_argument(
        '--pop-size',
        default=32,
        type=int,
        dest='population_size',
        help='The number of individuals in the population.')
    parser.add_argument(
        '--layers',
        default='[8, 3]',
        type=lambda s: list(map(int, s[1:-1].split(','))),
        help='The layers of the Neural Network.')
    parser.add_argument(
        '-G',
        '--max-gen',
        default=1000,
        type=int,
        dest='max_generation',
        help='The maximum number of generations.')
    parser.add_argument(
        '-F',
        '--target-fitness',
        default=10000,
        type=int,
        dest='target_fitness',
        help='The goal fitness. When attained, the learning stops.')
    parser.add_argument(
        '-Se',
        '--selection-elitism',
        default=True,
        type=bool,
        dest='selection_elitism',
        help='Whether to always keep the best chromosome in the next '
             'generation. For the moment, this will increase the population '
             'size by 1.')
    parser.add_argument(
        '-Sp',
        '--selection-rank-prob',
        default=0.2,
        type=int,
        dest='selection_rank_prob',
        help='The probability used in rank selection.')
    parser.add_argument(
        '-Sk',
        '--selection-frac-to-keep',
        default=.3,
        type=float,
        dest='selection_keep_frac',
        help='The fraction of the population to keep during evolution.')
    parser.add_argument(
        '-Mp',
        '--mutation-prob',
        default=.8,
        type=float,
        dest='mutation_prob',
        help='The probability for a chromosome to mutate.')
    parser.add_argument(
        '-Mip',
        '--mutation-inner-prob',
        default=.1,
        type=float,
        dest='mutation_inner_prob',
        help='If the chromosome mutates, this is the probability for a gene '
             'to mutate.')
    parser.add_argument(
        '-Cp',
        '--crossover-prob',
        default=.8,
        type=float,
        dest='crossover_prob',
        help='The probability for two parent chromosomes to crossover.')
    parser.add_argument(
        '-Cup',
        '--crossover-uniform-prob',
        default=.25,
        type=float,
        dest='crossover_uniform_prob',
        help='The probability of a swap during an Uniform crossover.')
    parser.add_argument(
        '-Rpg',
        '--num-new-random-per-generation',
        default=8,
        type=int,
        dest='num_new_random_per_generation',
        help='The number Rpg of new random chromosomes to introduce at each '
             'generation. The total population will be pop-size + Rpg.')

    return parser.parse_args()


def main():
    args = parse_args()
    set_up_logging(args)

    genetic_controller = GeneticController(args)

    genetic_controller.run_simulation()


if __name__ == "__main__":
    main()
