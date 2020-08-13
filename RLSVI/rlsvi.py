# import tensorflow as tf
import numpy as np
import numpy.random as rn

# Choosing appropriate matrix multiplication function
# if tf.__version__ == '0.10.0':
#     mmul = tf.mul
# else:
#     mmul = tf.matmul
#
# mu = tf.placeholder(tf.float32, [None])
# covmat = tf.placeholder(tf.float32, [None, None])
# eps = tf.placeholder(tf.float32, [None])
# mvn_transform_op = tf.reshape(mmul(covmat, tf.reshape(eps, [-1, 1])), [-1]) + mu


# def mvn_draw(sess, m, s):
#     return sess.run(mvn_transform_op, feed_dict={mu: m, covmat: s, eps: rn.normal(0, 1, m.size)})


class bayesian_regressor:
    # Fit a bayesian least squares regression model with a Gaussian prior
    # and get the estimated coefficients and the covariance
    def __init__(self, k, sigma=1.0, lmb=1.0):
        self.X = []
        self.y = []
        self.k = k
        self.sigma = sigma
        self.sigmasq = sigma * sigma
        self.lmb = lmb
        self.theta_est = np.zeros([self.k], dtype=np.float32)
        self.cov_est = self.sigmasq * np.eye(self.k, dtype=np.float32)
        self.theta_sample = np.dot(self.cov_est, rn.normal(0, 1, k)) + self.theta_est
        # Tensorflow ops for computing the fit and covariance in estimate
        self.A = np.zeros((self.k, self.k), dtype=np.float32)
        self.b = np.zeros(self.k, dtype=np.float32)


    # Store given data to be fit later
    def add_data(self, x, yi):
        self.X.append(x)
        self.y.append(yi)

    # Reset stored data to start a new round of fitting
    def reset_data(self):
        self.X = []
        self.y = []

    # Perform fit
    def fit(self):
        if len(self.X) == 0:
            return
        self.cov_est = np.linalg.inv(np.matmul(self.A.T, self.A) / self.sigmasq + self.lmb * np.eye(self.k, self.k))
        self.theta_est = np.matmul(self.cov, np.matmul(self.A.T, np.reshape(self.b, [-1, 1])) / self.sigmasq)
        self.theta_est = self.theta_est.flatten()

    # Sample and store a set of parameters from the learnt Gaussian posterior
    def sample_from_posterior(self):
        pass
        # o.theta_sample = mvn_draw(sess, o.theta_est, o.cov_est)

    # Get regression output for given input
    def evaluate(self, x):
        return np.dot(self.theta_sample, x)


class rlsvi:
    # Stationary RLSVI algorithm implementation
    def __init__(self, ns, na, sigma=1.0, lmb=1.0, gamma=0.99):
        self.sigma = sigma
        self.sigmasq = sigma * sigma
        self.lmb = lmb
        self.gam = gamma
        self.ns = ns
        self.na = na
        # One regression model for each action
        self.models = [bayesian_regressor(ns, self.sigma, self.lmb) for i in range(self.na)]

    # Compute the estimate and reset the data stored in the models for a
    # new episode.
    def new_episode(self):
        for i in range(self.na):
            self.models[i].fit()
            self.models[i].sample_from_posterior()
            self.models[i].reset_data()

    # Return the best action according to current estimates for given state
    def choose_action(self, s):
        return np.argmax(([self.models[i].evaluate(s) for i in range(self.na)]))

    # Save state transition and reward data
    def add_data(self, s1, a, r, s2, done):
        # Compute the targets
        if done:
            bi = r
        else:
            bi = r + self.gam * np.max(([self.models[i].evaluate(s2) for i in range(self.na)]))
        # Store in model for the given action
        self.models[a].add_data(s1, bi)
