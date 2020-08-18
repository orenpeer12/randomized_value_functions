
import numpy as np

' Greedy policy w.r.t linear approximation of Q function '


class Policy:
    def __init__(self, feature_selection, weights_init=None):
        self.feature_selection = feature_selection
        self.actions = np.arange(self.feature_selection.num_actions)    # assume actions are [0,...,N-1]
        self.weights_dim = feature_selection.feature_dim

        if weights_init is None:
            self.weights = np.random.uniform(-1.0, 1.0, self.weights_dim)
        else:
            self.weights = weights_init

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights

    def q_values(self, state):
        q_values = []
        # compute estimated Q-values for all actions - assume actions are [0,...,N-1]
        for action in range(self.feature_selection.num_actions):
            features = self.feature_selection.evaluate(state, action)
            q_value = np.dot(self.weights, features)
            q_values.append(q_value)

        return np.array(q_values)

    def get_action(self, state):
        q_values = self.q_values(state)
        return np.argmax(q_values)
