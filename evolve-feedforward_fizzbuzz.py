"""
5-bit FizzBuzz example.

Inspired by http://joelgrus.com/2016/05/23/fizz-buzz-in-tensorflow/

"""

from __future__ import print_function
import os
import neat
import visualize

import random
from operator import itemgetter
import math

def binary_encode(i, num_digits):
    return tuple([i >> d & 1 for d in range(num_digits)])

def fizz_buzz_encode(i):
    if   i % 15 == 0: return (0, 0, 0, 1)
    elif i % 5  == 0: return (0, 0, 1, 0)
    elif i % 3  == 0: return (0, 1, 0, 0)
    else:             return (1, 0, 0, 0)

# This simple example only cares about divisibility by 3.  The rest we'll add later.
def fizz_buzz_encode(i):
    if i % 3  == 0: return (1,)
    else:             return (0,)

def fizz_buzz(i, prediction):
    #return [str(i), "fizz", "buzz", "fizzbuzz"][prediction.index(max(prediction))]
    return [str(i), 'fizz'][int(round(prediction[0]))]

bitLength = 5

#fizzbuzz_integers = [random.randint(0,127) for i in range(64)]
fizzbuzz_integers = [i for i in range(2**bitLength)]
fizzbuzz_inputs = [binary_encode(i,bitLength) for i in fizzbuzz_integers]
fizzbuzz_outputs = [fizz_buzz_encode(i) for i in fizzbuzz_integers]

# A random assortment of

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        exampleLength = float(len(fizzbuzz_inputs))
        genome.fitness = 1.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for xi, xo, integer in zip(fizzbuzz_inputs, fizzbuzz_outputs, fizzbuzz_integers):
            output = net.activate(xi)

            guess = output[0]
            answer = xo[0]

            if integer < 8:
                penalty = 2**3.0
            else:
                penalty = float(2**(2.0*math.floor(math.log(integer,2)+1.0)-3.0-1.0))

            # This fitness function is scaled to give higher weights to smaller numbers.
            # Any number below 8 has a penalty of 1/8.
            # Any number above 8 has a penalty of 1/(2*binary encoding length - 4).
            #   If the program is correct in 0 through 7, it gets 1 point.
            #   If the program is correct in 8 through 15, it gets 1/2 point.
            #   If the program is correct in 16 through 31, it gets 1/4 point.
            # The total fitness will always range between 0 and 1.

            genome.fitness -= (guess - answer) ** 2 / penalty

def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
     neat.DefaultSpeciesSet, neat.DefaultStagnation,
     config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1000-1))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 5000)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for xi, xo in zip(fizzbuzz_inputs, fizzbuzz_outputs):
        output = winner_net.activate(xi)
        print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

    outputList = [fizz_buzz(i, winner_net.activate(binary_encode(i,bitLength))) for i in range(2**bitLength)];

    print(outputList)

    node_names = {-1:'A', -2: 'B', -3: 'C', -4: 'D', -5: 'E', -6: 'F', -7: 'G', 0:'Number', 1: 'fizz', 2: 'buzz', 3: 'fizzbuzz'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    #visualize.plot_species(stats, view=True)

    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    #p.run(eval_genomes, 10)




if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward-fizzbuzz')
    run(config_path)
