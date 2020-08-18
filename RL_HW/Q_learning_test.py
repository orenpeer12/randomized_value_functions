
from RL_HW.Q_learning import *
from RL_HW.memory import Buffer
from RL_HW.mountain_car_with_data_collection import MountainCarWithResetEnv
from RL_HW.test_model import test_model

# Parameters
BUFFER_SIZE = 100000
NUM_BASIS_FUNC = 4
GAMMA = 0.999
TEST_EXPERIMENTS = 10
LEARNING_RATE = 0.001
UPDATE_EVERY = 25

if __name__ == '__main__':
    # Initialize environment
    env = MountainCarWithResetEnv()
    state_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n

    # Initialize buffer and fill with random transitions, just to get mean and std
    buffer = Buffer(maxlen=BUFFER_SIZE)
    buffer.fill(env)

    # extract mean, std of data in buffer
    mean, std = buffer.compute_statistics()

    # Construct Q learning agent
    agent = Q_learning(buffer_size=BUFFER_SIZE, state_dim=state_dim, num_actions=num_actions,
                       gamma=GAMMA, lr=LEARNING_RATE, update_every=UPDATE_EVERY, num_basis_func=NUM_BASIS_FUNC,
                       mean=mean, std=std, low=env.low, high=env.high)

    # Train Q learning agent
    agent.train(env)
    success_rate = test_model(env, agent, num_initial_states=TEST_EXPERIMENTS)
    print('Success rate is: ' + str(success_rate))

