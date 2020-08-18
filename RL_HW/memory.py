
# Imports
import random
import numpy as np
from collections import namedtuple


Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


class Buffer(object):

    def __init__(self, maxlen=100000):
        self.buffer_size = maxlen
        self.buffer = []
        self.position = 0

    def insert(self, transition):
        """ Saves a transition """
        if len(self.buffer) < self.buffer_size:    # case buffer not full
            self.buffer.append(transition)
        else:                               # case buffer is full
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.buffer_size

    def sample(self, batch_size):
        transition_batch = random.sample(self.buffer, batch_size)
        return transition_batch

    def fill(self, env):    # TODO: Note that data collection is random!
        # unpack environment parameters
        min_pos, min_speed = env.low.tolist()
        max_pos, max_speed = env.high.tolist()

        for idx in range(self.buffer_size):
            # draw random initial state
            rand_pos = np.random.uniform(min_pos, max_pos)
            rand_speed = np.random.uniform(min_speed, max_speed)
            state = np.array([rand_pos, rand_speed])
            # reset environment to specific random state
            env.reset_specific(state[0], state[1])
            # draw random action
            action = env.action_space.sample()
            # act on environment
            next_state, reward, done, _ = env.step(action)
            # pack as transition
            transition = Transition(state=state, action=action, reward=reward,
                                    next_state=next_state, done=int(done))
            # insert to buffer
            self.insert(transition)

    def compute_statistics(self):
        # extract states from buffer
        states = [transition.state for transition in self.buffer]
        # compute mean and std of states
        mean, std = np.mean(states, axis=0), np.std(states, axis=0)
        # # normalize states and next states in buffer
        # for transtion_idx in range(len(self.buffer)):
        #     self.buffer[transtion_idx].state[:] -= mean
        #     self.buffer[transtion_idx].state[:] /= std
        #     self.buffer[transtion_idx].next_state[:] -= mean
        #     self.buffer[transtion_idx].next_state[:] /= std
        return mean, std

    def __len__(self):
        return len(self.buffer)




