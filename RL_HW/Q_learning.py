from RL_HW.feature_selection import RBF, StateAction2Features
from RL_HW.policy import Policy
from RL_HW.memory import *


class Q_learning:
    def __init__(self, buffer_size=100000, state_dim=2, num_actions=3, gamma=0.999, lr=1e-3,
                 update_every=25, num_basis_func=4, mean=0, std=1., low=None, high=None):

        self.memory = Buffer(maxlen=buffer_size)    # initialize buffer

        if low is None or high is None:
            low = np.array([-1., 1.])
            high = np.array([-1., 1.])

        self.basis_func = RBF(state_dim, num_basis_func, beta=1, low=low, high=high)
        self.feature_selection = StateAction2Features(num_actions, mean, std, self.basis_func)

        self.policy = Policy(self.feature_selection)

        self.update_every = update_every
        self.epsilon = 0.1      # epsilon-greedy parameter (exploration)
        self.stopping_criteria = 1e-6      # stopping criteria
        self.max_iter = 1e5    # maximum number of Q iterations
        self.gamma = gamma
        self.lr = lr

    # TODO: batch update
    def train(self, env, num_episodes=400, max_steps_per_episode=300):
        w = self.policy.get_weights()
        for episode in range(num_episodes):
            env.reset()
            for step in range(max_steps_per_episode):
                # collect data from environment to buffer
                self.collect_data(env, num_transitions=self.update_every)
                # get current transition from buffer
                transition = self.memory.buffer[self.memory.position - 1]
                # get next action according to greedy policy
                next_action = self.policy.get_action(transition.next_state)

                # get features of (s, a) and (s', a*)
                phi = self.feature_selection.evaluate(transition.state, transition.action)
                if transition.done:
                    phi_next = 0.
                    env.reset()     # if episode is over, reset env
                else:
                    phi_next = self.feature_selection.evaluate(transition.next_state, next_action)

                # Approximate TD error
                td_error = transition.reward + self.gamma * np.dot(phi_next, w) - np.dot(phi, w)
                # update weights
                w = w + self.lr * td_error * phi
                # update policy
                self.policy.set_weights(w)
            if (episode + 1) % 5 == 0:
                print(str(episode + 1) + ' episodes completed!')
                # print
                # print('Iteration: ' + str(num_iter) + ' -- Difference between weights: '
                #       + str(np.linalg.norm(w_next - w).round(5)))

    def collect_data(self, env, num_transitions=1):
        def epsilon_greedy(state, policy, epsilon):
            if np.random.uniform() <= epsilon:
                action = env.action_space.sample()
            else:
                action = policy.get_action(state)
            return action

        # get state from environment
        state = env.state
        for transition in range(num_transitions):
            # get action with epsilon-greedy exploration
            action = epsilon_greedy(state, self.policy, self.epsilon)
            # act on environment
            next_state, reward, done, _ = env.step(action)
            # pack as transition
            transition = Transition(state=state, action=action, reward=reward,
                                    next_state=next_state, done=int(done))
            # insert to buffer
            self.memory.insert(transition)
            # update current state
            state = next_state
