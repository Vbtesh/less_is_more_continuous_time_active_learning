
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

# Implementation of a OU Network which can be either fed realised data about trajectories and interventions to be plotted or be given a model and parameters to realise and then plot

class OU_Network():
    def __init__(self, N, K, true_model, theta, dt, sigma, init_state=None, data=None, interventions=None, range_values=(-100, 100), labels=['one', 'two', 'three']):
        # Parameters
        if len(true_model.shape) > 1:
            self._G = true_model
        else:
            self._G = self._causality_matrix(true_model)

        self._sig = sigma
        self._dt = dt
        self._theta = theta
        self._range = range_values

        # Visual info
        self._labels = labels
    
        # State initialisation
        self._N = N         # End of trial, maximum datapoints
        self._n = 0         # Current index in trial
        self._K = K
        self._data_history = []  # List that stores realisations of the network, parameters cannot be changed
        self._inter_history = [] # List that stores interventions on the network, follows the same indexing as data_history

        if type(data) == np.ndarray:
            self._X = data
            self._N = data.shape[0]
            self._n = self._N
            if type(interventions) == np.ndarray:
                self._I = interventions # Intervention must be the same length as data and contain the indexes corresponding to the variables intervened upon
            else:
                self._I = np.empty(self._N)
                self._I[:] = np.nan

        else:
            self._X = np.zeros((N+1, K))
            # If state is non initial, set the first row of the X matrix to be the initial state
            if type(init_state) == np.ndarray:
                self._X[0,:] = init_state

            # Initialise array of empty interventions
            self._I = np.empty(self._N+1)
            self._I[:] = np.nan
            
    
    def run(self, iter=1, interventions=None, reset=False):
        if reset:
            self.reset(save=True) # Store last run in history
            
        if self._n > self._N:
            print('Iterations maxed out')
            return
        elif iter > self._N - self._n:
            r_iter = self._N - self._n
        else:
            r_iter = iter

        # Run iterations
        for i in range(r_iter):
            if type(interventions) == np.ndarray:
                if len(interventions.shape) > 1:
                    intervention = interventions[i,:]
                else:
                    intervention = interventions
            else:
                intervention = None
            
            self.update(intervention) # Update the network

        # Return the generated values
        return self._X[self._n-r_iter+1:self._n+1, :]


    def update(self, intervention=None):
        # Compute attractor
        self_attractor = self._X[self._n,:] * (1 - np.abs(self._X[self._n,:]) / np.max(self._range))
        causal_attractor = self._X[self._n,:] @ self._G
        att = self_attractor + causal_attractor

        # Update using a direct sample from a normal distribution
        self._X[self._n+1, :] = np.random.normal(loc=self._X[self._n,:] + self._theta * self._dt * (att - self._X[self._n,:]), scale=self._sig*np.sqrt(self._dt)) 

        # If intervention, set value irrespective of causal matrix
        if type(intervention) == np.ndarray and np.sum(np.isnan(intervention)) == 0:
            inter_var = int(intervention[0])
            inter_val = intervention[1]
            self._X[self._n+1, inter_var] = inter_val
            self._I[self._n+1] = inter_var

        # Bound values
        self._X[self._n+1, :][self._X[self._n+1, :] < self._range[0]] = self._range[0]
        self._X[self._n+1, :][self._X[self._n+1, :] > self._range[1]] = self._range[1]

        # Increment index      
        self._n += 1


    def reset(self, rollback=0, save=True):
        if save:
            self._data_history.append(self._X[0:self._n+1,:]) # Save data in history
            self._inter_history.append(self._X[0:self._n+1])  # Save interventions in history
        self._n -= rollback      # Reset iter index
        self._X[self._n+1:,:] = 0 # Reset data except for the initial state
        # Reset interventions
        self._I[self._n+1:] = np.nan


    def plot_network(self, sns, history=None):
        palette = sns.color_palette() # Set palette
        sns.set_palette(palette)

        for i in range(self._K):
            # PLot data 
            ax = sns.lineplot(data=self._X[0:self._n+1,i], lw=1.5, label=self._labels[i]) # Plot data

            # Plot interventions where relevant
            ints = self._I[0:self._n+1] == i
            if np.sum(ints) == 0:
                continue
            
            x = np.arange(len(ints))
            y1 = self._range[0] * ints
            y2 = self._range[1] * ints
            ax.fill_between(x, y1, y2, color=palette[i], alpha=0.15)

        plt.title('Network realisation')
        plt.legend()
        plt.ylim(self._range[0], self._range[1])

        # Plot history
        if history:
            pass


    # Properties
    @property
    def N(self):
        return self._N

    @property
    def K(self):
        return self._K
    
    @property
    def causal_vector(self):
        return self._causality_vector(self._G)

    @property
    def causal_matrix(self):
        return self._G

    @causal_matrix.setter
    def causal_matrix(self, model):
        self._G = model

    @property
    def sigma(self):
        return self._sig
    
    @property
    def theta(self):
        return self._theta   

    @property
    def data(self):
        return self._X[0:self._n+1,:] 

    @property
    def data_last(self):
        return self._X[self._n,:]

    @property
    def inters(self):
        return self._I[0:self._n+1]

    @property
    def history(self):
        return (self._data_history, self._inter_history)


    # Internal methods
    def _causality_matrix(self, link_vec, fill_diag=0):
        num_var = int((1 + np.sqrt(1 + 4*len(link_vec))) / 2)
        causal_mat = fill_diag * np.ones((num_var, num_var))

        idx = 0
        for i in range(num_var):
            for j in range(num_var):
                if i != j:
                    causal_mat[i, j] = link_vec[idx] 
                    idx += 1

        return causal_mat

    def _causality_vector(self, link_mat):
        s = link_mat.shape[0]**2 - link_mat.shape[0]

        causal_vec = np.zeros(s)

        idx = 0
        for i in range(link_mat.shape[0]):
            for j in range(link_mat.shape[0]):
                if i != j:
                    causal_vec[idx] = link_mat[i, j]
                    idx += 1

        return causal_vec
