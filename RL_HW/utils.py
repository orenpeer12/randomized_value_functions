
import numpy as np
import matplotlib.pyplot as plt


def plot_states(states):
    plt.figure()
    states_arr = np.vstack(states)
    plt.scatter(states_arr[:, 0], states_arr[:, 1], marker='.')

    plt.grid('on')
    # get min and max to limit axes
    low = np.min(states_arr, axis=0)
    high = np.max(states_arr, axis=0)
    plt.xlim([1.1 * low[0], 1.1 * high[0]])
    plt.ylim([1.1 * low[1], 1.1 * high[1]])
    plt.xlabel('position')
    plt.ylabel('velocity')
    plt.title('Visualization of states in 2D')

