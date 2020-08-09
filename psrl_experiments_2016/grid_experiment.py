'''
Script to run tabular experiments in batch mode.

author: iosband@stanford.edu
'''

import numpy as np
import pandas as pd
import argparse
import sys

from src import environment
from src import finite_tabular_agents

from src.feature_extractor import FeatureTrueState
from src.experiment import run_finite_tabular_experiment



if __name__ == '__main__':
    '''
    Run a tabular experiment according to command line arguments
    '''
    # Take in command line flags
    parser = argparse.ArgumentParser(description='Run tabular RL experiment')
    parser.add_argument('--gridSize', help='size of grid', type=int, default=10)
    parser.add_argument('--epLen', help='episode length', type=int, default=20)
    parser.add_argument('--rewardVar', help='reward variance', type=float, default=0)
    parser.add_argument('--pNoise', help='size of grid', type=float, default=0)
    parser.add_argument('--alg', help='Agent constructor', type=str, default="UCRL2")
    parser.add_argument('--scaling', help='scaling', type=float, default=0)
    parser.add_argument('--seed', help='random seed', type=int, default=1)
    parser.add_argument('--nEps', help='number of episodes', type=int, default=10001)
    args = parser.parse_args()

    # Make a filename to identify flags
    fileName = ('stochasticGrid'
                + '_size=' + '%03.f' % args.gridSize
                + '_epLen=' + '%03.f' % args.epLen
                + '_rewardVar=' + '%03.2f' % args.rewardVar
                + '_pNoise=' + '%.3f' % args.pNoise
                + '_alg=' + str(args.alg)
                + '_scal=' + '%03.2f' % args.scaling
                + '_seed=' + str(args.seed)
                + '.csv')

    folderName = './'
    targetPath = folderName + fileName
    print('******************************************************************')
    print(fileName)
    print('******************************************************************')

    # Make the environment
    env = environment.make_stochasticGrid(args.gridSize,args.epLen,args.pNoise,args.rewardVar)

    # Make the feature extractor
    f_ext = FeatureTrueState(env.epLen, env.nState, env.nAction, env.nState)

    # Make the agent
    alg_dict = {'PSRL': finite_tabular_agents.PSRL,
                'PSRLunif': finite_tabular_agents.PSRLunif,
                'OptimisticPSRL': finite_tabular_agents.OptimisticPSRL,
                'GaussianPSRL': finite_tabular_agents.GaussianPSRL,
                'UCBVI': finite_tabular_agents.UCBVI,
                'BEB': finite_tabular_agents.BEB,
                'BOLT': finite_tabular_agents.BOLT,
                'UCRL2': finite_tabular_agents.UCRL2,
                'UCRL2_GP': finite_tabular_agents.UCRL2_GP,
                'UCRL2_GP_RTDP': finite_tabular_agents.UCRL2_GP_RTDP,
                'EULER': finite_tabular_agents.EULER,
                'EULER_GP': finite_tabular_agents.EULER_GP,
                'EULER_GP_RTDP': finite_tabular_agents.EULER_GP_RTDP,
                'UCFH': finite_tabular_agents.UCFH,
                'EpsilonGreedy': finite_tabular_agents.EpsilonGreedy}

    agent_constructor = alg_dict[args.alg]

    agent = agent_constructor(env.nState, env.nAction, env.epLen,
                              scaling=args.scaling)

    # Run the experiment
    run_finite_tabular_experiment(agent, env, f_ext, args.nEps, args.seed,
                        recFreq=100, fileFreq=1000, targetPath=targetPath)

