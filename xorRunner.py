from __future__ import print_function
import os
import neat
import visualize
import random

examples = 80.0

def binary_encode(i, num_digits):
  return tuple([i >> d & 1 for d in range(num_digits)])

def generate_data():
  inputs = []
  outputs = []
  x = 0
  while x < examples:
    number = random.randint(1,127)
    inputs.append(binary_encode(number, 7))
    if (number % 3 == 0) and (number % 5 == 0):
      outputs.append((0.0,0.0,0.0,1.0))
    elif number % 3 == 0:
      outputs.append((0.0,1.0,0.0,0.0))
    elif number % 5 == 0:
      outputs.append((0.0,0.0,1.0,0.0))
    else:
      outputs.append((1.0,0.0,0.0,0.0))

    x += 1

  return [inputs, outputs]
  


data = generate_data()
xor_outputs = data[1]
xor_inputs = data[0]
  


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 1
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for xi, xo in zip(xor_inputs, xor_outputs):
            output = net.activate(xi)
            x = (output.index(max(output)) != xo.index(max(xo))) / examples
            genome.fitness -= x


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
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 10000)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = winner_net.activate(xi)
        print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

    node_names = {}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    p.run(eval_genomes, 100)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)