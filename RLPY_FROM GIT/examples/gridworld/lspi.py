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


def make_experiment(exp_id=1, path="./Results/Temp"):
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
    maze = os.path.join(GridWorld.default_map_dir, '11x11-Rooms.txt')
    # maze = os.path.join(GridWorld.default_map_dir, '10x10-12ftml.txt')
    domain = GridWorld(maze, noise=0.3)
    opt["domain"] = domain

    # Representation
    representation = Tabular(domain)

    # Policy
    policy = eGreedy(representation, epsilon=0.1)

    # Agent
    opt["agent"] = LSPI(policy, representation, domain.discount_factor,
                 opt["max_steps"], 1000)

    experiment = Experiment(**opt)
    return experiment

if __name__ == '__main__':
    path = "./Results/Temp/{domain}/{agent}/{representation}/"
    experiment = make_experiment(1, path=path)
    experiment.run(visualize_steps=True,  # should each learning step be shown?
                   visualize_learning=False,  # show performance runs?
                   visualize_performance=False)  # show value function?
    experiment.plot()
    experiment.save()
