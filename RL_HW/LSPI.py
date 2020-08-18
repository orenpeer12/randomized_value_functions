
import numpy as np
from RL_HW.feature_selection import RBF, StateAction2Features
from RL_HW.policy import Policy


class LSPI:
    def __init__(self, buffer, state_dim=2, num_actions=3, gamma=0.999, max_iter=10,
                 num_basis_func=4, mean=0, std=1., low=None, high=None):

        self.memory = buffer

        if low is None or high is None:
            low = np.array([-1., 1.])
            high = np.array([-1., 1.])

        self.basis_func = RBF(state_dim, num_basis_func, beta=1, low=low, high=high)
        self.feature_selection = StateAction2Features(num_actions, mean, std, self.basis_func)

        self.policy = Policy(self.feature_selection)
        self.lstdq = LSTDQ(self.feature_selection, self.policy, gamma)

        self.epsilon = 1e-3     # stopping criterion
        self.max_iter = max_iter    # maximum number of LSPI iterations
        self.gamma = gamma

    def train(self, batch_size):
        w = np.inf * np.ones(self.feature_selection.feature_dim)
        w_next = self.policy.get_weights()
        num_iter = 0    # bound max number of iterations

        while np.linalg.norm(w_next - w) > self.epsilon and num_iter < self.max_iter:
            # copy weights
            w = w_next.copy()
            # update policy
            self.policy.set_weights(w)
            # sample batch
            transition_batch = self.memory.sample(batch_size)
            # evaluate weights
            w_next = self.lstdq.fit_weights(transition_batch, self.policy)

            # update counter
            num_iter += 1

            # print
            print('Iteration: ' + str(num_iter) + ' -- Difference between weights: '
                  + str(np.linalg.norm(w_next - w).round(5)))


class LSTDQ:
    def __init__(self, feature_selection, init_policy, gamma):
        self.feature_selection = feature_selection
        self.gamma = gamma
        self.policy = init_policy

    def fit_weights(self, transition_batch, policy):
        k = self.feature_selection.feature_dim  # dimension of feature vector

        delta_C = 0.01  # constant to fill in diagonal -- stability
        C = np.zeros([k, k])
        d = np.zeros(k)
        np.fill_diagonal(C, delta_C)

        for transition in transition_batch:
            # action according to greedy policy
            next_action = policy.get_action(transition.next_state)
            # get features from (s, a) and (s', a*)
            phi = self.feature_selection.evaluate(transition.state, transition.action)
            # to handle episodic case - take phi_next=0 when done
            if transition.done:
                phi_next = 0.
            else:
                phi_next = self.feature_selection.evaluate(transition.next_state, next_action)

            # add element to estimation of C, d
            C = C + np.outer(phi, (phi - self.gamma * phi_next))
            d = d + phi * transition.reward
        # compute the inverse of matrix C
        inv_C = np.linalg.inv(C)
        # compute estimated weights
        weights = inv_C @ d

        return weights

