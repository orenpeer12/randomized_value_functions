
from RL_HW.LSPI import *
from RL_HW.memory import Buffer
from RL_HW.mountain_car_with_data_collection import MountainCarWithResetEnv
from RL_HW.test_model import test_model

# Parameters
BUFFER_SIZE = 100000
NUM_BASIS_FUNC = 4
GAMMA = 0.999
BATCH_SIZE = 1024 * 8
TEST_EXPERIMENTS = 10

# if __name__ == '__main__':
#     # Initialize environment
#     env = MountainCarWithResetEnv()
#     state_dim = env.observation_space.shape[0]
#     num_actions = env.action_space.n
#
#     # Initialize buffer
#     buffer = Buffer(maxlen=BUFFER_SIZE)
#
#     # Fill buffer with random transitions
#     buffer.fill(env)
#
#     # extract mean, std of data in buffer
#     mean, std = buffer.compute_statistics()
#
#     # Construct LSPI agent
#     agent = LSPI(buffer=buffer, state_dim=state_dim, num_actions=num_actions,
#                  gamma=GAMMA, num_basis_func=NUM_BASIS_FUNC, mean=mean, std=std,
#                  low=env.low, high=env.high)
#
#     # Train LSPI agent
#     agent.train(batch_size=BATCH_SIZE)
#     success_rate = test_model(env, agent, num_initial_states=TEST_EXPERIMENTS)
#     print('Success rate is: ' + str(success_rate))



if __name__ == '__main__':
    # Initialize environment
    env = MountainCarWithResetEnv()
    state_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n

    # Initialize buffer
    buffer = Buffer(maxlen=BUFFER_SIZE)

    # Fill buffer with random transitions
    buffer.fill(env)

    # extract mean, std of data in buffer
    mean, std = buffer.compute_statistics()

    # Construct LSPI agent
    agent = LSPI(buffer=buffer, state_dim=state_dim, num_actions=num_actions,
                 gamma=GAMMA, num_basis_func=NUM_BASIS_FUNC, mean=mean, std=std,
                 low=env.low, high=env.high)

    # Train LSPI agent
    agent.train(batch_size=BATCH_SIZE)
    success_rate = test_model(env, agent, num_initial_states=TEST_EXPERIMENTS)
    print('Success rate is: ' + str(success_rate))

