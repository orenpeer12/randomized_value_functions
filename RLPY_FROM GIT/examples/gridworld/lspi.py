#!/usr/bin/env python

from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library

standard_library.install_aliases()
__author__ = "William Dabney"

from rlpy.Domains import GridWorld
from rlpy.Agents import LSPI
from rlpy.Representations import Tabular
from rlpy.Policies import eGreedy
from rlpy.Experiments import Experiment
import os


def make_experiment(noise, exp_id=1, path="./Results/Temp"):
    """
    Each file specifying an experimental setup should contain a
    make_experiment function which returns an instance of the Experiment
    class with everything set up.

    @param id: number used to seed the random number generators
    @param path: output directory where logs and results are stored
    """

    # Experiment variables
    opt = {}
    opt["path"] = path
    opt["exp_id"] = exp_id
    opt["max_steps"] = 10000
    opt["num_policy_checks"] = 10
    opt["checks_per_policy"] = 50

    # Logging

    # Domain:
    # MAZE                = '/Domains/GridWorldMaps/1x3.txt'
    # maze = os.path.join(GridWorld.default_map_dir, '4x5.txt')
    # maze = os.path.join(GridWorld.default_map_dir, '6x9-Wall.txt')
    # maze = os.path.join(GridWorld.default_map_dir, '11x11-Rooms.txt')
    # maze = os.path.join(GridWorld.default_map_dir, '10x10-12ftml.txt')
    maze = os.path.join(GridWorld.default_map_dir, 'oren.txt')
    domain = GridWorld(maze, noise=0.0)
    opt["domain"] = domain

    # Representation
    representation = Tabular(domain)

    # Policy
    # policy = eGreedy(representation, epsilon=0)  # OREN
    policy = eGreedy(representation, epsilon=0.1)
    # policy = eGreedy(representation, epsilon=0.3)

    # Agent
    opt["agent"] = LSPI(noise, policy, representation, domain.discount_factor,
                        opt["max_steps"], 300)

    experiment = Experiment(**opt)
    return experiment


if __name__ == '__main__':
    # noise = 0.0000001
    # noise = 0.1
    noise = 0
    # noise = 5
    # title = "eps0.0"
    # title = "eps0.1"
    # title = "eps0.5"
    # title = "$\epsilon-greedy$ exploration, $\epsilon=0$"
    title = "$\epsilon-greedy$ exploration, $\epsilon=0.1$"
    # title = "b noise"
    path = "./Results/Temp/{domain}/{agent}/{representation}/"
    experiment = make_experiment(exp_id=1, path=path, noise=noise)
    experiment.run(visualize_steps=False,  # should each learning step be shown?
                   visualize_learning=False,  # show performance runs?
                   visualize_performance=False)  # show value function?
    experiment.plot(save=True, title=title)
    experiment.save()
