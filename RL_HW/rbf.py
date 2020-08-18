
import numpy as np


# class RBF:
#     def __init__(self, input_dim, num_basis_func, beta, low, high):
#         self.input_dim = input_dim
#         self.num_basis_func = num_basis_func
#         self.low = low
#         self.high = high
#         self.means = self.compute_means(num_basis_func)
#         self.beta = beta     # coefficient of the distance in the Gaussian exponent
#
#     def evaluate(self, s):
#         sqr_mean_dists = [np.sum((s - mean) ** 2) for mean in self.means]
#         features = [np.exp(-self.beta * sqr_mean_dist) for sqr_mean_dist in sqr_mean_dists]
#         return np.array(features)
#
#     def compute_means(self, num_basis_func):
#         if self.input_dim == 2:     # case 2-D (our case), choose uniform coverage of box
#             x_ticks = np.linspace(self.low[0], self.high[0], int(np.sqrt(num_basis_func)) + 2)[1:-1]
#             y_ticks = np.linspace(self.low[1], self.high[1], int(np.sqrt(num_basis_func)) + 2)[1:-1]
#             xv, yv = np.meshgrid(x_ticks, y_ticks, indexing='ij')
#             means = []
#             for i in range(len(x_ticks)):
#                 for j in range(len(y_ticks)):
#                     means.append(np.array([xv[i, j], yv[i, j]]))
#
#             # case that num_basis_func is not exact square root - add random means
#             if len(means) < num_basis_func:
#                 means.append([np.random.uniform(self.low, self.high, self.input_dim)
#                               for i in range(num_basis_func - len(means))])
#         else:
#             means = [np.random.uniform(self.low, self.high, self.input_dim) for i in range(num_basis_func)]
#
#         return means


def encode_states(states, rbf, mean, std):
    encoded_states = []
    for state in states:
        # standardize the state
        encoded_state = (state - mean) / std
        # encode using rbf
        encoded_state = rbf.evaluate(encoded_state)
        # add bias term
        # encoded_state = np.append(encoded_state, 1.)
        encoded_states.append(encoded_state)


