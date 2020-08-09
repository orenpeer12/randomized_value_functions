'''
Finite horizon tabular agents.

This is a collection of some of the classic benchmark algorithms for efficient
reinforcement learning in a tabular MDP with little/no prior knowledge.
We provide implementations of:

- PSRL
- Gaussian PSRL
- UCBVI
- BEB
- BOLT
- UCRL2
- Epsilon-greedy

author: iosband@stanford.edu
'''

import numpy as np
from src.agent import *
import copy

class FiniteHorizonTabularAgent(FiniteHorizonAgent):
    '''
    Simple tabular Bayesian learner from Tabula Rasa.

    Child agents will mainly implement:
        update_policy

    Important internal representation is given by qVals and qMax.
        qVals - qVals[state, timestep] is vector of Q values for each action
        qMax - qMax[timestep] is the vector of optimal values at timestep

    '''

    def __init__(self, nState, nAction, epLen,
                 alpha0=1., mu0=0., tau0=1., tau=1., **kwargs):
        '''
        Tabular episodic learner for time-homoegenous MDP.
        Must be used together with true state feature extractor.

        Args:
            nState - int - number of states
            nAction - int - number of actions
            alpha0 - prior weight for uniform Dirichlet
            mu0 - prior mean rewards
            tau0 - precision of prior mean rewards
            tau - precision of reward noise

        Returns:
            tabular learner, to be inherited from
        '''
        # Instantiate the Bayes learner
        self.nState = nState
        self.nAction = nAction
        self.epLen = epLen
        self.alpha0 = alpha0
        self.mu0 = mu0
        self.tau0 = tau0
        self.tau = tau

        self.qVals = {}
        self.qMax = {}

        # Now make the prior beliefs
        self.R_prior = {}
        self.P_prior = {}

        for state in range(nState):
            for action in range(nAction):
                self.R_prior[state, action] = (self.mu0, self.tau0)
                self.P_prior[state, action] = (
                    self.alpha0 * np.ones(self.nState, dtype=np.float32))

    def update_obs(self, oldState, action, reward, newState, pContinue, h):
        '''
        Update the posterior belief based on one transition.

        Args:
            oldState - int
            action - int
            reward - double
            newState - int
            pContinue - 0/1
            h - int - time within episode (not used)

        Returns:
            NULL - updates in place
        '''

        mu0, tau0 = self.R_prior[oldState, action]
        tau1 = tau0 + self.tau
        mu1 = (mu0 * tau0 + reward * self.tau) / tau1
        self.R_prior[oldState, action] = (mu1, tau1)

        if pContinue == 1:
            self.P_prior[oldState, action][newState] += 1

    def egreedy(self, state, timestep, epsilon=0):
        """
        Select action according to a greedy policy

        Args:
            state - int - current state
            timestep - int - timestep *within* episode

        Returns:
            action - int
        """
        Q = self.qVals[state, timestep]
        nAction = Q.size
        noise = np.random.rand()

        if noise < epsilon:
            action = np.random.choice(nAction)
        else:
            action = np.random.choice(np.where(Q == Q.max())[0])

        return action

    def pick_action(self, state, timestep,nEps=100):
        '''
        Default is to use egreedy for action selection
        '''
        action = self.egreedy(state, timestep)
        return action

    def sample_mdp(self):
        '''
        Returns a single sampled MDP from the posterior.

        Args:
            NULL

        Returns:
            R_samp - R_samp[s, a] is the sampled mean reward for (s,a)
            P_samp - P_samp[s, a] is the sampled transition vector for (s,a)
        '''
        R_samp = {}
        P_samp = {}
        for s in range(self.nState):
            for a in range(self.nAction):
                mu, tau = self.R_prior[s, a]
                R_samp[s, a] = mu + np.random.normal() * 1./np.sqrt(tau)
                P_samp[s, a] = np.random.dirichlet(self.P_prior[s, a])

        return R_samp, P_samp

    def map_mdp(self):
        '''
        Returns the maximum a posteriori MDP from the posterior.

        Args:
            NULL

        Returns:
            R_hat - R_hat[s, a] is the MAP mean reward for (s,a)
            P_hat - P_hat[s, a] is the MAP transition vector for (s,a)
        '''
        R_hat = {}
        P_hat = {}
        for s in range(self.nState):
            for a in range(self.nAction):
                R_hat[s, a] = self.R_prior[s, a][0]
                P_hat[s, a] = self.P_prior[s, a] / np.sum(self.P_prior[s, a])

        return R_hat, P_hat

    def compute_qVals(self, R, P):
        '''
        Compute the Q values for a given R, P estimates

        Args:
            R - R[s,a] = mean rewards
            P - P[s,a] = probability vector of transitions

        Returns:
            qVals - qVals[state, timestep] is vector of Q values for each action
            qMax - qMax[timestep] is the vector of optimal values at timestep
        '''
        qVals = {}
        qMax = {}

        qMax[self.epLen] = np.zeros(self.nState, dtype=np.float32)

        for i in range(self.epLen):
            j = self.epLen - i - 1
            qMax[j] = np.zeros(self.nState, dtype=np.float32)

            for s in range(self.nState):
                qVals[s, j] = np.zeros(self.nAction, dtype=np.float32)

                for a in range(self.nAction):
                    qVals[s, j][a] = R[s, a] + np.dot(P[s, a], qMax[j + 1])

                qMax[j][s] = np.max(qVals[s, j])

        return qVals, qMax

    def compute_qVals_opt(self, R, P, R_bonus, P_bonus):
        '''
        Compute the Q values for a given R, P estimates + R/P bonus

        Args:
            R - R[s,a] = mean rewards
            P - P[s,a] = probability vector of transitions
            R_bonus - R_bonus[s,a] = bonus for rewards
            P_bonus - P_bonus[s,a] = bonus for transitions

        Returns:
            qVals - qVals[state, timestep] is vector of Q values for each action
            qMax - qMax[timestep] is the vector of optimal values at timestep
        '''
        qVals = {}
        qMax = {}

        qMax[self.epLen] = np.zeros(self.nState, dtype=np.float32)

        for i in range(self.epLen):
            j = self.epLen - i - 1
            qMax[j] = np.zeros(self.nState, dtype=np.float32)

            for s in range(self.nState):
                qVals[s, j] = np.zeros(self.nAction, dtype=np.float32)

                for a in range(self.nAction):
                    qVals[s, j][a] = (R[s, a] + R_bonus[s, a]
                                      + np.dot(P[s, a], qMax[j + 1])
                                      + P_bonus[s, a] * i)
                qMax[j][s] = np.max(qVals[s, j])

        return qVals, qMax

    def compute_qVals_EVI(self, R, P, R_slack, P_slack, qMax = {}):
        '''
        Compute the Q values for a given R, P by extended value iteration

        Args:
            R - R[s,a] = mean rewards
            P - P[s,a] = probability vector of transitions
            R_slack - R_slack[s,a] = slack for rewards
            P_slack - P_slack[s,a] = slack for transitions
            qMax - Initial qMax values (taking the minimum between previous value and the new value)
                   It allows testing the effect of intersecting the confidence interval - leave empty to ignore

        Returns:
            qVals - qVals[state, timestep] is vector of Q values for each action
            qMax - qMax[timestep] is the vector of optimal values at timestep
        '''
        # Extended value iteration
        qVals = {}
        if not qMax: # no qMax was provided - we calculate everything from scratch
            calc_minimum = False
            qMax = {}
            qMax[self.epLen] = np.zeros(self.nState)
        else: # when updating qMax, we take the minimum between the new and previous value
            calc_minimum = True

        for i in range(self.epLen):
            j = self.epLen - i - 1
            if not calc_minimum:
                qMax[j] = np.zeros(self.nState)

            for s in range(self.nState):
                qVals[s, j] = np.zeros(self.nAction)

                for a in range(self.nAction):
                    rOpt = R[s, a] + R_slack[s, a]

                    # form pOpt by extended value iteration, pInd sorts the values
                    pInd = np.argsort(qMax[j + 1])
                    pOpt = np.copy(P[s, a])
                    if pOpt[pInd[self.nState - 1]] + P_slack[s, a] * 0.5 > 1:
                        pOpt = np.zeros(self.nState)
                        pOpt[pInd[self.nState - 1]] = 1
                    else:
                        pOpt[pInd[self.nState - 1]] += P_slack[s, a] * 0.5

                    # Go through all the states and get back to make pOpt a real prob
                    sLoop = 0
                    while np.sum(pOpt) > 1:
                        worst = pInd[sLoop]
                        pOpt[worst] = max(0, 1 - np.sum(pOpt) + pOpt[worst])
                        sLoop += 1

                    # Do Bellman backups with the optimistic R and P
                    qVals[s, j][a] = rOpt + np.dot(pOpt, qMax[j + 1])

                if calc_minimum:
                    qMax[j][s] = min(np.max(qVals[s, j]), qMax[j][s])
                else:
                    qMax[j][s] = np.max(qVals[s, j])

        return qVals, qMax

#-----------------------------------------------------------------------------
# PSRL
#-----------------------------------------------------------------------------

class PSRL(FiniteHorizonTabularAgent):
    '''
    Posterior Sampling for Reinforcement Learning
    '''

    def update_policy(self, h=False, nEps=False):
        '''
        Sample a single MDP from the posterior and solve for optimal Q values.

        Works in place with no arguments.
        '''
        # Sample the MDP
        R_samp, P_samp = self.sample_mdp()

        # Solve the MDP via value iteration
        qVals, qMax = self.compute_qVals(R_samp, P_samp)

        # Update the Agent's Q-values
        self.qVals = qVals
        self.qMax = qMax

#-----------------------------------------------------------------------------
# PSRL
#-----------------------------------------------------------------------------

class PSRLunif(PSRL):
    '''
    Posterior Sampling for Reinforcement Learning with spread prior
    '''

    def __init__(self, nState, nAction, epLen,
                 alpha0=1., mu0=0., tau0=1., tau=1., **kwargs):
        '''
        Just like PSRL but rescale alpha between successor states

        Args:
            nSamp - int - number of samples to use for optimism
        '''
        newAlpha = alpha0 / nState
        super(PSRLunif, self).__init__(nState, nAction, epLen, alpha0=newAlpha,
                                       mu0=mu0, tau0=tau0, tau=tau)

#-----------------------------------------------------------------------------
# Optimistic PSRL
#-----------------------------------------------------------------------------

class OptimisticPSRL(PSRL):
    '''
    Optimistic Posterior Sampling for Reinforcement Learning
    '''
    def __init__(self, nState, nAction, epLen,
                 alpha0=1., mu0=0., tau0=1., tau=1., nSamp=10, **kwargs):
        '''
        Just like PSRL but we take optimistic over multiple samples

        Args:
            nSamp - int - number of samples to use for optimism
        '''
        super(OptimisticPSRL, self).__init__(nState, nAction, epLen,
                                             alpha0, mu0, tau0, tau)
        self.nSamp = nSamp

    def update_policy(self, h=False, nEps=False):
        '''
        Take multiple samples and then take the optimistic envelope.

        Works in place with no arguments.
        '''
        # Sample the MDP
        R_samp, P_samp = self.sample_mdp()
        qVals, qMax = self.compute_qVals(R_samp, P_samp)
        self.qVals = qVals
        self.qMax = qMax

        for i in range(1, self.nSamp):
            # Do another sample and take optimistic Q-values
            R_samp, P_samp = self.sample_mdp()
            qVals, qMax = self.compute_qVals(R_samp, P_samp)

            for timestep in range(self.epLen):
                self.qMax[timestep] = np.maximum(qMax[timestep],
                                                 self.qMax[timestep])
                for state in range(self.nState):
                    self.qVals[state, timestep] = np.maximum(qVals[state, timestep],
                                                             self.qVals[state, timestep])

#-----------------------------------------------------------------------------
# Gaussian PSRL
#-----------------------------------------------------------------------------

class GaussianPSRL(FiniteHorizonTabularAgent):
    '''Naive Gaussian approximation to PSRL, similar to tabular RLSVI'''

    def __init__(self, nState, nAction, epLen,
                 alpha0=1., mu0=0., tau0=1., tau=1., scaling=1.):
        '''
        As per the tabular learner, but added tunable scaling.

        Args:
            scaling - double - rescale default confidence sets
        '''
        super(GaussianPSRL, self).__init__(nState, nAction, epLen, alpha0,
                                    mu0, tau0, tau)
        self.scaling = scaling

    def gen_bonus(self, h=False):
        ''' Generate the Gaussian bonus for Gaussian PSRL '''
        R_bonus = {}
        P_bonus = {}
        for s in range(self.nState):
            for a in range(self.nAction):
                R_sum = self.R_prior[s, a][1]
                R_bonus[s, a] = self.scaling * np.random.normal() * 1. / np.sqrt(R_sum)

                P_sum = self.P_prior[s, a].sum()
                P_bonus[s, a] = self.scaling * np.random.normal() * 1. / np.sqrt(P_sum)

        return R_bonus, P_bonus

    def update_policy(self, h=False, nEps=False):
        '''
        Update Q values via Gaussian PSRL.
        This performs value iteration but with additive Gaussian noise.
        '''
        # Output the MAP estimate MDP
        R_hat, P_hat = self.map_mdp()

        # Purely Gaussian perturbations
        R_bonus, P_bonus = self.gen_bonus(h)

        # Form approximate Q-value estimates
        qVals, qMax = self.compute_qVals_opt(R_hat, P_hat, R_bonus, P_bonus)

        self.qVals = qVals
        self.qMax = qMax

#-----------------------------------------------------------------------------
# UCBVI
#-----------------------------------------------------------------------------

class UCBVI(GaussianPSRL):
    '''Upper confidence bounds value iteration... similar to Gaussian PSRL'''

    def gen_bonus(self, h=1):
        ''' Generate the sqrt(n) bonus for UCBVI '''
        R_bonus = {}
        P_bonus = {}
        for s in range(self.nState):
            for a in range(self.nAction):
                R_sum = self.R_prior[s, a][1]
                R_bonus[s, a] = self.scaling * np.sqrt(2. * np.log(2 + h) / R_sum)

                P_sum = self.P_prior[s, a].sum()
                P_bonus[s, a] = self.scaling * np.sqrt(2. * np.log(2 + h) / P_sum)

        return R_bonus, P_bonus

#-----------------------------------------------------------------------------
# BEB
#-----------------------------------------------------------------------------

class BEB(GaussianPSRL):
    '''BayesExploreBonus BEB algorithm'''

    def __init__(self, nState, nAction, epLen,
                 alpha0=1., mu0=0., tau0=1., tau=1., scaling=1.):
        '''
        As per the tabular learner, but added tunable scaling.

        Args:
            scaling - double - rescale default confidence sets
        '''
        super(BEB, self).__init__(nState, nAction, epLen,
                                                alpha0, mu0, tau0, tau)
        self.beta = 2 * self.epLen * self.epLen * scaling

    def gen_bonus(self, h=False):
        ''' Generate the 1/n bonus for BEB'''
        R_bonus = {}
        P_bonus = {}
        for s in range(self.nState):
            for a in range(self.nAction):
                R_sum = self.R_prior[s, a][1]
                R_bonus[s, a] = 1. / (R_sum + 1)

                P_sum = self.P_prior[s, a].sum()
                P_bonus[s, a] = self.beta * self.epLen / (1 + P_sum)

        return R_bonus, P_bonus

#-----------------------------------------------------------------------------
# BOLT
#-----------------------------------------------------------------------------

class BOLT(FiniteHorizonTabularAgent):
    '''Bayes Optimistic Local Transitions (BOLT)'''

    def __init__(self, nState, nAction, epLen,
                 alpha0=1., mu0=0., tau0=1., tau=1., scaling=1.):
        '''
        As per the tabular learner, but added tunable scaling.

        Args:
            scaling - double - rescale default confidence sets
        '''
        super(BOLT, self).__init__(nState, nAction, epLen,
                                    alpha0, mu0, tau0, tau)
        self.eta = self.epLen * scaling

    def get_slack(self, time):
        '''
        Returns the slackness parameters for BOLT.
        These are based upon eta imagined optimistic observations

        Args:
            time - int - grows the confidence sets

        Returns:
            R_slack - R_slack[s, a] is the confidence width for BOLT reward
            P_slack - P_slack[s, a] is the confidence width for BOLT transition
        '''
        R_slack = {}
        P_slack = {}

        for s in range(self.nState):
            for a in range(self.nAction):
                R_slack[s, a] = self.eta / (self.R_prior[s, a][1] + self.eta)
                P_slack[s, a] = 2 * self.eta / (self.P_prior[s, a].sum() + self.eta)
        return R_slack, P_slack

    def update_policy(self, h=False, nEps=False):
        '''
        Compute BOLT Q-values via extended value iteration.
        '''
        # Output the MAP estimate MDP
        R_hat, P_hat = self.map_mdp()

        # Compute the slack parameters
        R_slack, P_slack = self.get_slack(h)

        # Perform extended value iteration
        qVals, qMax = self.compute_qVals_EVI(R_hat, P_hat, R_slack, P_slack)

        self.qVals = qVals
        self.qMax = qMax

#-----------------------------------------------------------------------------
# UCRL2
#-----------------------------------------------------------------------------

class UCRL2(FiniteHorizonTabularAgent):
    '''Classic benchmark optimistic algorithm'''

    def __init__(self, nState, nAction, epLen,
                 delta=0.05, scaling=1., **kwargs):
        '''
        As per the tabular learner, but prior effect --> 0.

        Args:
            delta - double - probability scale parameter
            scaling - double - rescale default confidence sets
        '''
        super(UCRL2, self).__init__(nState, nAction, epLen,
                                    alpha0=1e-5, tau0=0.0001)
        self.delta = delta
        self.scaling = scaling

        # optimistic initialization
        self.qVals = {}
        self.qMax = {}
        for h in range(epLen + 1):
            self.qMax[h] = (epLen - h) * np.ones(nState)
            if h < self.epLen + 1:
                for s in range(nState):
                    for a in range(nAction):
                        self.qVals[s, h] = (epLen - h) * np.ones(nAction)


    def get_slack(self, time):
        '''
        Returns the slackness parameters for UCRL2

        Args:
            time - int - grows the confidence sets

        Returns:
            R_slack - R_slack[s, a] is the confidence width for UCRL2 reward
            P_slack - P_slack[s, a] is the confidence width for UCRL2 transition
        '''
        R_slack = {}
        P_slack = {}
        delta = self.delta
        scaling = self.scaling
        for s in range(self.nState):
            for a in range(self.nAction):
                nObsR = max(self.R_prior[s, a][1] - self.tau0, 1.)
                R_slack[s, a] = scaling * np.sqrt((2 * np.log(6 * self.nState * self.nAction * (time+1) / delta)) / float(nObsR))

                nObsP = max(self.P_prior[s, a].sum() - self.alpha0*self.nState, 1.)
                P_slack[s, a] = scaling * np.sqrt((4 * self.nState * np.log(9 * self.nState * self.nAction * (time + 1) / delta)) / float(nObsP))
        return R_slack, P_slack

    def update_policy(self, time=100,nEps=100):
        '''
        Compute UCRL2 Q-values via extended value iteration.
        '''
        # Output the MAP estimate MDP
        R_hat, P_hat = self.map_mdp()

        # Compute the slack parameters
        R_slack, P_slack = self.get_slack(nEps*self.epLen)

        # Perform extended value iteration
        qVals, qMax = self.compute_qVals_EVI(R_hat, P_hat, R_slack, P_slack)

        self.qVals = qVals
        self.qMax = qMax

#-----------------------------------------------------------------------------
# UCRL2_GP
#-----------------------------------------------------------------------------

class UCRL2_GP(UCRL2):
    '''Efroni+Merlis modifications to UCRL2 for RTDP
    This implementation mostly keeps the modules' functionality as other algorithms, but less efficient than
    UCRL2_GP_RTDP'''

    def __init__(self, nState, nAction, epLen,
                 delta=0.05, scaling=1., **kwargs):
        '''
        As per the tabular learner, but prior effect --> 0.

        Args:
            delta - double - probability scale parameter
            scaling - double - rescale default confidence sets
        '''
        # We use smaller effective delta, due to the additional union bounds in comparison to UCRL2
        super(UCRL2_GP, self).__init__(nState, nAction, epLen,delta*3/4,scaling)

        # optimistic initialization
        self.qVals = {}
        self.qMax = {}
        for h in range(epLen+1):
            self.qMax[h] = (epLen - h) * np.ones(nState)
            if h < self.epLen + 1:
                for s in range(nState):
                    for a in range(nAction):
                        self.qVals[s,h] = (epLen-h)*np.ones(nAction)

        # We need  to save the values from the previous iteration, so that we update only visited states:
        self.qMax_new = copy.deepcopy(self.qMax)


    def update_obs(self, oldState, action, reward, newState, pContinue, h):
        '''
        Update the posterior belief based on one transition.

        Args:
            oldState - int
            action - int
            reward - double
            newState - int
            pContinue - 0/1
            h - int - time within episode (not used)

        Returns:
            NULL - updates in place
        '''

        mu0, tau0 = self.R_prior[oldState, action]
        tau1 = tau0 + self.tau
        mu1 = (mu0 * tau0 + reward * self.tau) / tau1
        self.R_prior[oldState, action] = (mu1, tau1)

        if pContinue == 1:
            self.P_prior[oldState, action][newState] += 1

        # Update the Q value for the specific state we visited:
        self.qMax_new[h][oldState] = self.qMax[h][oldState]

    def update_policy(self, time=100,nEps=100):
        '''
        Updates the policy with Forward-pass (similarly to RTDP)
        In this function, we update all of the states. Later, we will copy only the visited states in update_obs
        '''
        # First - copy the Q-values that were actually updated from the previous iteration
        self.qMax = copy.deepcopy(self.qMax_new)

        # Output the MAP estimate MDP
        R_hat, P_hat = self.map_mdp()

        # Compute the slack parameters
        R_slack, P_slack = self.get_slack(nEps*self.epLen)

        # Perform 'forward' extended value iteration

        for i in range(self.epLen):
            for s in range(self.nState):
                for a in range(self.nAction):
                    rOpt = R_hat[s, a] + R_slack[s, a]

                    # form pOpt by extended value iteration, pInd sorts the values
                    pInd = np.argsort(self.qMax[i + 1])
                    pOpt = np.copy(P_hat[s, a])
                    if pOpt[pInd[self.nState - 1]] + P_slack[s, a] * 0.5 >= 1:
                        pOpt = np.zeros(self.nState)
                        pOpt[pInd[self.nState - 1]] = 1
                    else:
                        pOpt[pInd[self.nState - 1]] += P_slack[s, a] * 0.5

                    # Go through all the states and get back to make pOpt a real prob
                    sLoop = 0
                    while np.sum(pOpt) > 1:
                        worst = pInd[sLoop]
                        pOpt[worst] = max(0, 1 - np.sum(pOpt) + pOpt[worst])
                        sLoop += 1

                    # Do Bellman backups with the optimistic R and P
                    self.qVals[s, i][a] = rOpt + np.dot(pOpt, self.qMax[i + 1])

                self.qMax[i][s] = min(np.max(self.qVals[s, i]),self.qMax[i][s])

#-----------------------------------------------------------------------------
# UCRL2_GP_RTDP
#-----------------------------------------------------------------------------

class UCRL2_GP_RTDP(UCRL2):
    '''Efroni+Merlis modifications to UCRL2 for RTDP
    This implementation directly implements RTDP, and 'bypasses' the code natural structure, but is more efficient.'''

    def __init__(self, nState, nAction, epLen,
                 delta=0.05, scaling=1., **kwargs):
        '''
        As per the tabular learner, but prior effect --> 0.

        Args:
            delta - double - probability scale parameter
            scaling - double - rescale default confidence sets
        '''
        # We use smaller effective delta, due to the additional union bound in comparison to UCRL2
        super(UCRL2_GP_RTDP, self).__init__(nState, nAction, epLen,delta*3/4,scaling)

        # optimistic initialization
        self.qVals = {}
        self.qMax = {}
        for h in range(epLen+1):
            self.qMax[h] = (epLen - h) * np.ones(nState)
            if h < self.epLen + 1:
                for s in range(nState):
                    for a in range(nAction):
                        self.qVals[s,h] = (epLen-h)*np.ones(nAction)

        # We need  to save the visitations from the previous iteration, so we don't take into account the state
        # visitation in the current episode when updating the value
        self.R_prior_new = copy.deepcopy(self.R_prior)
        self.P_prior_new = copy.deepcopy(self.P_prior)


    def pick_action(self, state, timestep, nEps):

        '''
        Updates the policy with Forward-pass (similarly to RTDP)
        '''

        # Output the MAP estimate MDP
        R_hat, P_hat = self.map_mdp()

        # Compute the slack parameters
        R_slack, P_slack = self.get_slack(nEps * self.epLen)

        # Perform 'forward' extended value iteration step
        for a in range(self.nAction):
            rOpt = R_hat[state, a] + R_slack[state, a]

            # form pOpt by extended value iteration, pInd sorts the values
            pInd = np.argsort(self.qMax[timestep + 1])
            pOpt = np.copy(P_hat[state, a])
            if pOpt[pInd[self.nState - 1]] + P_slack[state, a] * 0.5 >= 1:
                pOpt = np.zeros(self.nState)
                pOpt[pInd[self.nState - 1]] = 1
            else:
                pOpt[pInd[self.nState - 1]] += P_slack[state, a] * 0.5

            # Go through all the states and get back to make pOpt a real prob
            sLoop = 0
            while np.sum(pOpt) > 1:
                worst = pInd[sLoop]
                pOpt[worst] = max(0, 1 - np.sum(pOpt) + pOpt[worst])
                sLoop += 1

            # Do Bellman backups with the optimistic R and P
            self.qVals[state, timestep][a] = rOpt + np.dot(pOpt, self.qMax[timestep + 1])

        self.qMax[timestep][state] = min(np.max(self.qVals[state, timestep]), self.qMax[timestep][state])


        '''
        Default is to use egreedy for action selection
        '''
        action = self.egreedy(state, timestep)
        return action

    def update_obs(self, oldState, action, reward, newState, pContinue, h):
        '''
        Update the posterior belief based on one transition.

        Args:
            oldState - int
            action - int
            reward - double
            newState - int
            pContinue - 0/1
            h - int - time within episode (not used)

        Returns:
            NULL - updates in place
        '''

        mu0, tau0 = self.R_prior_new[oldState, action]
        tau1 = tau0 + self.tau
        mu1 = (mu0 * tau0 + reward * self.tau) / tau1
        self.R_prior_new[oldState, action] = (mu1, tau1)

        if pContinue == 1:
            self.P_prior_new[oldState, action][newState] += 1


    def update_policy(self, time=100,nEps=100):
        '''
        Save the counts \ empirical estimates copies
        '''

        # Update the counts
        self.R_prior = copy.deepcopy(self.R_prior_new)
        self.P_prior = copy.deepcopy(self.P_prior_new)



#-----------------------------------------------------------------------------
# EULER
#-----------------------------------------------------------------------------

class EULER(FiniteHorizonTabularAgent):
    '''EULER by Zanette and Brunskill, 2019'''

    def __init__(self, nState, nAction, epLen,
                 delta=0.05, scaling=1., **kwargs):
        '''
        As per the tabular learner, but prior effect --> 0.

        Args:
            delta - double - probability scale parameter
            scaling - double - rescale default confidence sets
        '''
        super(EULER, self).__init__(nState, nAction, epLen,
                                    alpha0=1e-5, tau0=0.0001)
        self.delta = delta
        self.scaling = scaling
        self.qMax = {}  # lower bound on the values
        self.qMin = {} # lower bound on the values

        for h in range(epLen+1):
                self.qMax[h] = (epLen - h) * np.ones(nState)
                self.qMin[h] = np.zeros(nState)

        # We need the squared values for the variance calculations:
        self.R_squared_sum = {}

        for state in range(nState):
            for action in range(nAction):
                self.R_squared_sum[state, action] = 0


    def get_slack(self, time,h):
        '''
        Returns the slackness parameters for UCRL2

        Args:
            time - int - grows the confidence sets
            h - int - the timestep inside the episode

        Returns:
            R_slack - R_slack[s, a] is the confidence width for EULER reward
            NextVal_slack - P_slack[s, a] is the confidence width for EULER transition+value
        '''

        # Output the MAP estimate MDP
        R_hat, P_hat = self.map_mdp()

        R_slack = {}
        NextVal_slack = {}
        delta = self.delta
        delta0 = delta/7
        scaling = self.scaling
        L = np.log(4 * self.nState * self.nAction * time / delta0) # log term
        # doing the value iteration + optimism for EULER
        for s in range(self.nState):
            for a in range(self.nAction):
                # reward counts
                nObsR = self.R_prior[s, a][1] - self.tau0
                nObsR_sat = max(nObsR, 1.)
                nObsR_minus1_sat = max(nObsR-1, 1.)

                # calculated the unbiased reward variance estimator
                R_variance = (self.R_squared_sum[s,a] -nObsR*R_hat[s,a]**2) / nObsR_minus1_sat # unbiased variance estimator

                # reward optimism
                R_slack[s, a] = scaling * (np.sqrt( 2*R_variance*L / float(nObsR_sat) ) + 14*L/(3*nObsR_sat))

                # transition counts
                nObsP = self.P_prior[s, a].sum() - self.alpha0*self.nState
                nObsP_sat = max(nObsP, 1.)

                # calculating the value variance w.r.t. the estimated transtion kernel
                V_Variance = np.dot(P_hat[s,a],(self.qMax[h+1] - np.dot(P_hat[s,a],self.qMax[h+1]))**2)

                # calculating the the weighted norm of the difference between the upper and lower bounds of the value
                delta_V_norm = np.sqrt(np.dot(P_hat[s,a],(self.qMax[h+1]-self.qMin[h+1])**2))

                # calculating all the total confidence interval on p^T*V
                NextVal_slack[s, a] = scaling * (np.sqrt(2*V_Variance*L/nObsP_sat) + 2*self.epLen*L/(3*nObsP_sat) +  #phi(s,a)
                                                 self.epLen*(8*L/3 + np.sqrt(2*L))/nObsP_sat + np.sqrt(2*L)*delta_V_norm/np.sqrt(nObsP_sat) )
        return R_slack, NextVal_slack

    def update_obs(self, oldState, action, reward, newState, pContinue, h):
        '''
        Update the posterior belief based on one transition.

        Args:
            oldState - int
            action - int
            reward - double
            newState - int
            pContinue - 0/1
            h - int - time within episode (not used)

        Returns:
            NULL - updates in place
        '''

        mu0, tau0 = self.R_prior[oldState, action]
        tau1 = tau0 + self.tau
        mu1 = (mu0 * tau0 + reward * self.tau) / tau1
        self.R_prior[oldState, action] = (mu1, tau1)

        if pContinue == 1:
            self.P_prior[oldState, action][newState] += 1

        self.R_squared_sum[oldState,action] += reward**2

    def update_policy(self, time=100,nEps=100):
        '''
        Compute EULER Q-values via value iteration.
        '''
        # Output the MAP estimate MDP
        R_hat, P_hat = self.map_mdp()

        # Value iteration for EULER
        qVals = {}
        qMax = {}
        qMin = {}
        qMax[self.epLen] = np.zeros(self.nState)
        qMin[self.epLen] = np.zeros(self.nState)

        for i in range(self.epLen):
            j = self.epLen - i - 1
            qMax[j] = np.zeros(self.nState)
            qMin[j] = np.zeros(self.nState)

            R_slack, NextVal_slack = self.get_slack(nEps*self.epLen,j)

            for s in range(self.nState):
                qVals[s, j] = np.zeros(self.nAction)

                for a in range(self.nAction):
                    rOpt = R_hat[s, a] + R_slack[s, a]

                    NextValOpt = np.dot(P_hat[s, a],qMax[j + 1]) + NextVal_slack[s,a] # optimistic p^T*V

                    # Do Bellman backups with the optimistic R and next value
                    qVals[s, j][a] = rOpt + NextValOpt

                best_action= np.argmax(qVals[s, j])
                qMax[j][s] = min(qVals[s,j][best_action],self.epLen-j)

                NextValPass = R_hat[s, best_action] - R_slack[s, best_action] + \
                              np.dot(P_hat[s, best_action],qMin[j + 1])- NextVal_slack[s,best_action] # passimistic
                qMin[j][s] = max(NextValPass,0)

        self.qVals = qVals
        self.qMax = qMax
        self.qMin = qMin


#-----------------------------------------------------------------------------
# EULER_GP
#-----------------------------------------------------------------------------

class EULER_GP(EULER):
    '''Efroni+Merlis modifications to EULER for RTDP
    This implementation mostly keeps the modules' functionality as other algorithms, but less efficient than
    EULER_GP_RTDP'''

    def __init__(self, nState, nAction, epLen,
                 delta=0.05, scaling=1., **kwargs):
        '''
        As per the tabular learner, but prior effect --> 0.

        Args:
            delta - double - probability scale parameter
            scaling - double - rescale default confidence sets
        '''
        # We use smaller effective delta, due to the additional union bound in comparison to EULER
        super(EULER_GP, self).__init__(nState, nAction, epLen,delta*7/9, scaling)

        # We need  to save the values from the previous iteration, so that we update only visited states:
        self.qMax_new = copy.deepcopy(self.qMax)
        self.qMin_new = copy.deepcopy(self.qMin)



    def update_obs(self, oldState, action, reward, newState, pContinue, h):
        '''
        Update the posterior belief based on one transition.

        Args:
            oldState - int
            action - int
            reward - double
            newState - int
            pContinue - 0/1
            h - int - time within episode (not used)

        Returns:
            NULL - updates in place
        '''

        mu0, tau0 = self.R_prior[oldState, action]
        tau1 = tau0 + self.tau
        mu1 = (mu0 * tau0 + reward * self.tau) / tau1
        self.R_prior[oldState, action] = (mu1, tau1)

        if pContinue == 1:
            self.P_prior[oldState, action][newState] += 1

        self.R_squared_sum[oldState,action] += reward**2

        # Update the values for the specific state we visited:
        self.qMax_new[h][oldState] = self.qMax[h][oldState]
        self.qMin_new[h][oldState] = self.qMin[h][oldState]

    def update_policy(self, time=100,nEps=100):
        '''
        Compute EULER Q-values via value iteration.
        In this function, we update all of the states. Later, we will copy only the visited states in update_obs.
        '''

        # First - copy the values that were actually updated from the previous iteration
        self.qMax = copy.deepcopy(self.qMax_new)
        self.qMin = copy.deepcopy(self.qMin_new)
        qVals = {}

        # Output the MAP estimate MDP
        R_hat, P_hat = self.map_mdp()

        # Value iteration for EULER
        for i in range(self.epLen):

            R_slack, NextVal_slack = self.get_slack(nEps*self.epLen,i)

            for s in range(self.nState):
                qVals[s, i] = np.zeros(self.nAction)
                for a in range(self.nAction):
                    rOpt = R_hat[s, a] + R_slack[s, a]

                    NextValOpt = np.dot(P_hat[s, a],self.qMax[i + 1]) + NextVal_slack[s,a] # optimistic p^T*V

                    # Do Bellman backups with the optimistic R and next value
                    qVals[s, i][a] = rOpt + NextValOpt

                best_action= np.argmax(qVals[s, i])
                self.qMax[i][s] = min(qVals[s,i][best_action],self.qMax[i][s])

                NextValPass = R_hat[s, best_action] - R_slack[s, best_action] + \
                              np.dot(P_hat[s, best_action],self.qMin[i + 1])- NextVal_slack[s,best_action] # passimistic
                self.qMin[i][s] = max(NextValPass,self.qMin[i][s])

        self.qVals = qVals


#-----------------------------------------------------------------------------
# EULER_GP_RTDP
#-----------------------------------------------------------------------------

class EULER_GP_RTDP(EULER):
    '''Efroni+Merlis modifications to EULER for RTDP
    This implementation directly implements RTDP, and 'bypasses' the code natural structure, but is more efficient.'''

    def __init__(self, nState, nAction, epLen,
                 delta=0.05, scaling=1., **kwargs):
        '''
        As per the tabular learner, but prior effect --> 0.

        Args:
            delta - double - probability scale parameter
            scaling - double - rescale default confidence sets
        '''
        # We use smaller effective delta, due to the additional union bound in comparison to EULER
        super(EULER_GP_RTDP, self).__init__(nState, nAction, epLen,delta*7/9, scaling)

        # We need  to save the visitations from the previous iteration, so we don't take into account the state
        # visitation in the current episode when updating the value
        self.R_prior_new = copy.deepcopy(self.R_prior)
        self.P_prior_new = copy.deepcopy(self.P_prior)
        self.R_squared_sum_new = copy.deepcopy(self.R_squared_sum)


    def update_obs(self, oldState, action, reward, newState, pContinue, h):
        '''
        Update the posterior belief based on one transition.

        Args:
            oldState - int
            action - int
            reward - double
            newState - int
            pContinue - 0/1
            h - int - time within episode (not used)

        Returns:
            NULL - updates in place
        '''

        mu0, tau0 = self.R_prior_new[oldState, action]
        tau1 = tau0 + self.tau
        mu1 = (mu0 * tau0 + reward * self.tau) / tau1
        self.R_prior_new[oldState, action] = (mu1, tau1)

        if pContinue == 1:
            self.P_prior_new[oldState, action][newState] += 1

        self.R_squared_sum_new[oldState,action] += reward**2


    def pick_action(self, state, timestep, nEps):

        '''
        Compute EULER Q-values via value iteration.
        '''

        # Output the MAP estimate MDP
        R_hat, P_hat = self.map_mdp()

        # Value iteration for EULER
        R_slack, NextVal_slack = self.get_slack(nEps*self.epLen,timestep)

        self.qVals[state, timestep] = np.zeros(self.nAction)
        for a in range(self.nAction):
            rOpt = R_hat[state, a] + R_slack[state, a]

            NextValOpt = np.dot(P_hat[state, a],self.qMax[timestep + 1]) + NextVal_slack[state,a] # optimistic p^T*V

            # Do Bellman backups with the optimistic R and next value
            self.qVals[state, timestep][a] = rOpt + NextValOpt

        best_action= np.argmax(self.qVals[state, timestep])
        self.qMax[timestep][state] = min(self.qVals[state, timestep][best_action],self.qMax[timestep][state])

        NextValPass = R_hat[state, best_action] - R_slack[state, best_action] + \
                      np.dot(P_hat[state, best_action],self.qMin[timestep + 1])- NextVal_slack[state,best_action] # passimistic
        self.qMin[timestep][state] = max(NextValPass,self.qMin[timestep][state])

        # Default is to use egreedy for action selection
        action = self.egreedy(state, timestep)
        return action


    def update_policy(self, time=100,nEps=100):
        '''
        Save the counts \ empirical estimates copies
        '''

        # Update the counts
        self.R_prior = copy.deepcopy(self.R_prior_new)
        self.P_prior = copy.deepcopy(self.P_prior_new)
        self.R_squared_sum = copy.deepcopy(self.R_squared_sum_new)

#-----------------------------------------------------------------------------
# UCFH
#-----------------------------------------------------------------------------

class UCFH(UCRL2):
    '''Dann+Brunskill modificaitons to UCRL2 for finite domains'''

    def __init__(self, nState, nAction, epLen,
                 delta=0.05, scaling=1., epsilon=0.1, **kwargs):
        '''
        As per the tabular learner, but prior effect --> 0.

        Args:
            delta - double - probability scale parameter
            scaling - double - rescale default confidence sets
        '''
        super(UCFH, self).__init__(nState, nAction, epLen,
                                   alpha0=1e-9, tau0=0.0001)
        self.epsilon = epsilon
        self.delta = delta
        self.scaling = scaling
        self.epsilon = epsilon
        wMin = epsilon / (4 * nState * epLen)
        uMax = nState * nAction * np.log(nState * epLen / wMin) / np.log(2)
        self.delta1 = delta / (2 * uMax * nState)

    def compute_confidence(self, pHat, n):
        '''
        Compute the confidence sets for a give p component.
        Dann + Brunskill style

        Args:
            pHat - estimated transition probaility component
            n - number of observations
            delta - confidence paramters

        Returns:
            valid_p
        '''
        delta1 = self.delta1
        scaling = self.scaling
        target_sd = np.sqrt(pHat * (1 - pHat))
        K_1 = scaling * np.sqrt(2 * np.log(6 / delta1) / float(max(n - 1, 1)))
        K_2 = scaling * target_sd * K_1 + 7 / (3 * float(max(n - 1, 1))) * np.log(6 / delta1)

        sd_min = target_sd - K_1
        C_1 = (target_sd - K_1) * (target_sd - K_1)
        varLower, varUpper = (0, 1)

        # Only look after one side of variance inequality since Dann+Brunskill
        # algorithm ignores the other side anyway
        if sd_min > 1e-5 and C_1 > 0.2499:
            varLower = 0.5 * (1 - np.sqrt(1 - 4 * C_1))
            varUpper = 0.5 * (1 + np.sqrt(1 - 4 * C_1))

        # Empirical mean constrains
        mean_min = pHat - K_2
        mean_max = pHat + K_2

        # Checking the type of contstraint
        if pHat < varLower or pHat > varUpper:
            varLower, varUpper = (0, 1)

        # Don't worry about non-convex interval, since it is not used in paper
        interval = [np.max([0, varLower, mean_min]),
                    np.min([1, varUpper, mean_max])]
        return interval


    def update_policy(self, time=100,nEps=100):
        '''
        Updates the policy with UCFH extended value iteration
        '''
        # Extended value iteration
        qVals = {}
        qMax = {}
        qMax[self.epLen] = np.zeros(self.nState)

        # Output the MAP estimate MDP
        R_hat, P_hat = self.map_mdp()

        # Compute the slackness for rewards UCRL2 style
        R_slack = {}
        delta = self.delta
        delta1 = self.delta1
        scaling = self.scaling
        for s in range(self.nState):
            for a in range(self.nAction):
                nObsR = max(self.R_prior[s, a][1] - self.tau0, 1.)
                R_slack[s, a] = scaling * np.sqrt((4 * np.log(2 * self.nState * self.nAction * (time + 1) / delta)) / nObsR)

        P_range = {}
        # Extended value iteration as per Dann+Brunskill
        for i in range(self.epLen):
            j = self.epLen - i - 1
            qMax[j] = np.zeros(self.nState)

            for s in range(self.nState):
                qVals[s, j] = np.zeros(self.nAction)

                for a in range(self.nAction):
                    nObsP = max(self.P_prior[s, a].sum() - self.alpha0, 1.)
                    rOpt = R_hat[s, a] + R_slack[s, a]
                    pOpt = np.zeros(self.nState)

                    # pInd sorts the next-step values in *increasing* order
                    pInd = np.argsort(qMax[j + 1])

                    for sPrime in range(self.nState):
                        P_range[s, a, sPrime] = self.compute_confidence(P_hat[s,a][sPrime], nObsP)
                        pOpt[sPrime] = P_range[s, a, sPrime][0]

                    pSlack = 1 - pOpt.sum()

                    if pSlack < 0:
                        print('ERROR we have a problem')

                    for sPrime in range(self.nState):
                        # Reverse the ordering
                        newState = pInd[self.nState - sPrime - 1]
                        newSlack = min([pSlack, P_range[s, a, newState][1] - pOpt[newState]])
                        pOpt[newState] += newSlack
                        pSlack -= newSlack
                        if pSlack < 0.001:
                            break
                    qVals[s, j][a] = rOpt + np.dot(pOpt, qMax[j + 1])

                qMax[j][s] = np.max(qVals[s, j])
        self.qVals = qVals
        self.qMax = qMax

#-----------------------------------------------------------------------------
# Epsilon-Greedy
#-----------------------------------------------------------------------------

class EpsilonGreedy(FiniteHorizonTabularAgent):
    '''Epsilon greedy agent'''

    def __init__(self, nState, nAction, epLen, epsilon=0.1, **kwargs):
        '''
        As per the tabular learner, but prior effect --> 0.

        Args:
            epsilon - double - probability of random action
        '''
        super(EpsilonGreedy, self).__init__(nState, nAction, epLen,
                                            alpha0=0.0001, tau0=0.0001)
        self.epsilon = epsilon

    def update_policy(self, time=False, nEps=False):
        '''
        Compute UCRL Q-values via extended value iteration.

        Args:
            time - int - grows the confidence sets
        '''
        # Output the MAP estimate MDP
        R_hat, P_hat = self.map_mdp()

        # Solve the MDP via value iteration
        qVals, qMax = self.compute_qVals(R_hat, P_hat)

        # Update the Agent's Q-values
        self.qVals = qVals
        self.qMax = qMax

    def pick_action(self, state, timestep):
        '''
        Default is to use egreedy for action selection
        '''
        action = self.egreedy(state, timestep, self.epsilon)
        return action
