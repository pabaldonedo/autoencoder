#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import sys
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.linalg.blas import dgemm

# global variables for progress bar
total = point = increment = bar_pos = 0

class Sparse:
    def __init__(self, nodes=25, lamb=1e-4, rho=0.01, beta=3):
    	'''
        Object used for representing a Sparse network.

    	Parameters
		----------
		nodes : int
		    number of nodes in the hidden layer.

		lamb : float
		    backpropagation regularization parameter.
		    
		rho : float
		    sparsity parameter.
		    
		beta : float
		    lregularization parameter.

    	'''
        self.rho = float(rho)
        self.lamb = lamb
        self.beta = beta
        self.nodes = nodes

class Theta(object):
    '''
    Object used for representing Theta and bias terms.
    '''
    def __init__(self, sparse):
        self.num_input_nodes = sparse.inputs
        self.num_hidden_nodes = sparse.nodes

    def initialize_params(self, bound = 0.15):
        '''
        Initializes both theta and bias values randomly using a uniform distribution.
        '''
        self.theta1 = np.random.uniform(low=-bound, high=bound, size=(self.num_hidden_nodes, self.num_input_nodes))
        self.theta2 = np.random.uniform(low=-bound, high=bound, size=(self.num_input_nodes, self.num_hidden_nodes))
        self.b1 = np.random.uniform(low=-bound, high=bound, size=(self.num_hidden_nodes, 1))
        self.b2 = np.random.uniform(low=-bound, high=bound, size=(self.num_input_nodes, 1))

        return self.theta1, self.theta2, self.b1, self.b2

    def wrap_theta(self, x, thetaparam):
        self.theta1, self.theta2, self.b1, self.b2 = wrap_matrices(x, thetaparam)

    def unwrap_theta(self, thetaparam): #theta param list
        return unwrap_matrices(thetaparam)
        
class Cost(object):
    '''
    This class contains all methods involved in the Cost function calculation. 
    '''
    def __init__(self):
        self.derivative = None
        self.x = None
    
    def __call__(self, theta, *args):
        self.theta = theta
        cost, derivative = self.calculate(theta, *args)
        self.derivative = derivative
        return cost

    def calculate_derivative(self, x, *args):
        if self.derivative is not None and np.alltrue(x == self.x):
            return self.derivative
        else:
            self(x, *args)
            return self.derivative

    def calculate(self, theta, data, sparse):
        '''
        Calculates the cost function of a Sparse network for a given data using
        feedforward and backpropagation algorithms. 

        Parameters
        ----------
        theta : Theta
            Theta values of the Sparse network.

        data : numpy.ndarray
            Training data.

        sparse : Sparse
            Sparse network.

        Returns
        -------
        cost_function : float
            Value of the cost function.

        total_partials : numpy.ndarray
            Matrix containing theta and bias values.
        '''
        th1, th2, b1, b2 = wrap_matrices(sparse, theta)
        num_samples = data.shape[0]

        # layer 1
        a1 = data

        # layer 2
        bias2 = np.dot(np.ones((num_samples,1)),b1.T)
        z2 = np.dot(th1, data.T).T + bias2
        a2 = sigmoid(z2)

        # layer 3
        bias3 = np.dot(np.ones((num_samples,1)),b2.T)
        z3 = np.dot(th2, a2.T).T + bias3
        a3 = sigmoid(z3)

        ### feedforward cost ###
        reg = np.sum(np.power(th1, 2)) + np.sum(np.power(th2, 2))
        norm = np.power(a3 - data, 2)
        Jnn = 0.5 * np.sum(norm) / num_samples + 0.5 * sparse.lamb * reg

        rhoest = np.sum(a2, axis=0) / num_samples
        kl = self.__compute_kl(sparse.rho, rhoest)
        cost_function = Jnn + sparse.beta * np.sum(kl)
           
        ### Backpropagation ##
        sum2 = sparse.beta * (-sparse.rho / rhoest + (1.0 - sparse.rho) / (1.0 - rhoest))
        delta3 = (-(data - a3) * a3 * (1 - a3)).T
        sum1 = dgemm(alpha=1.0, a=th2.T, b=delta3, trans_b=False)
        sum2 = sum2[:,np.newaxis]
        delta2 = np.add(sum1,sum2)* a2.T * (1 - a2.T)

        partial_j1 = np.dot(delta2, a1)
        partial_j2 = np.dot(delta3, a2)
        partial_b1 = delta2.sum(axis=1)
        partial_b2 = delta3.sum(axis=1)
        total_partials = [1.0 / num_samples * partial_j1 + sparse.lamb * th1, 1.0 / num_samples * partial_j2 + sparse.lamb * th2, 1.0 / num_samples * partial_b1, 1.0 / num_samples * partial_b2]    
        return cost_function, unwrap_matrices(total_partials)

    def __compute_kl(self, rho, rhoest):
        return rho * np.log(rho / rhoest) + (1.0 - rho) * np.log((1.0 - rho) / (1.0 - rhoest))

def fit(sparse, data, epochs = 100, normalization = True, init = True, bound = 0.15):
    '''
    Fit the Sparse network supplied according to the given training data. 

    Parameters
    ----------
    sparse : Sparse
        Empty Sparse network.

    data : numpy.ndarray
        Training data.
    '''
    init_progressbar(epochs)
    sparse.inputs = data.shape[1]
    if init is True:
        sparse.theta1, sparse.theta2, sparse.b1, sparse.b2 = Theta(sparse).initialize_params(bound)

    if normalization:
        data, sparse.mean_axis, sparse.std = normalize(data)

    cost_function = Cost()
    bar = range(epochs)
    
    opt_theta, _, _ = fmin_l_bfgs_b(cost_function.calculate, unwrap_matrices([sparse.theta1, sparse.theta2, sparse.b1, sparse.b2]), 
        cost_function.calculate_derivative, args=(data, sparse), maxiter = epochs, iprint = 0, approx_grad = False, bounds = None, callback = display_progressbar)
    sparse.theta1, sparse.theta2, sparse.b1, sparse.b2 = wrap_matrices(sparse, opt_theta)
    print

def normalize(data, truncate_val = 3.):
    '''
    Normalization is done by first truncating data to +- x times its 
    standard deviation and then moving data to interval [0.1, 0.9].

    Parameters
    ----------
    data : numpy.ndarray
        Array containing input data.

    truncate_val : float
        truncate value (as times of standard deviations)

    Returns
    -------
    data : numpy.ndarray
        Normalized data array.

    mean_axis : numpy.ndarray
        Array containing the mean of each axis.

    data_std : numpy.ndarray
        Array containing the standard deviation of each axis.

    '''
    mean_axis = np.mean(data, axis = 0)
    data_std = np.std(data[:,:])
    data = data - mean_axis

    a_min = - truncate_val * data_std
    a_max = truncate_val * data_std
    data = np.clip(data, a_min, a_max) / a_max

    data = (data + 1) * 0.4 + 0.1 

    return data, mean_axis, data_std

def sigmoid(x):
    '''
    Returns the value of the sigmoid function evaluated in x.

    Parameters
    ----------
    data : numpy.ndarray
        Array containing data to be evaluated by the sigmoid function.

    Returns
    -------
    sigmoid_values : numpy.ndarray
        Points evaluated.
    '''
    return 1.0 / (1 + np.exp(-x))

def wrap_matrices(x, thetaparam):
    '''
    Wraps an array of four flatten matrices into four matrices.
    '''
    layer1 = thetaparam[0:x.nodes * x.inputs].reshape((x.nodes, x.inputs), order=1)
    index1 = x.nodes * x.inputs * 2
    layer2 = thetaparam[x.nodes * x.inputs:index1].reshape((x.inputs, x.nodes),order=1)
    index2 = index1 + x.nodes
    bias1 = thetaparam[index1:index2].reshape(x.nodes,1)
    bias2 = thetaparam[index2:].reshape(x.inputs,1)
    return layer1, layer2, bias1, bias2

def unwrap_matrices(thetaparam):
    th = np.empty(0)
    for i in thetaparam:
        th = np.hstack((th, i.flatten(1)))
    return th

def init_progressbar(x):
    '''
    Function that initializes the global variables used for 
    controling the plotting of the progress bar.

    Parameters
    ----------
    x : int
        Maximum number of iteration correspoding with 100% of the progressbar.
    '''
    global total, point, increment, bar_pos
    total = x
    point = total / 100
    increment = total / 40
    bar_pos = 0

def display_progressbar(xk):
    '''
    Function that updates the prompt with the new value of the progressbar. It 
    is used as a callback by fmin_l_bfgs_b whenever it finishes each iteration.

    Parameters
    ----------
    xk : numpy.ndarray
        Is the current parameter vector of fmin_l_bfgs_b.
    '''
    global total, point, increment, bar_pos
    if bar_pos % (2 * point) == 0:
        bar_chars = "=" * (bar_pos / increment)
        bar_empty = " " * ((total - bar_pos) / increment)
        bar_perc = bar_pos / point
        sys.stdout.write("\rTraining dataset [{0}{1}]{2}%".format(bar_chars, bar_empty, bar_perc))
        sys.stdout.flush()
    bar_pos = bar_pos + 1

def evaluate(sparse, data):
    '''
    Function that takes a trained sparse autoencoder network and some data and
    outputs the feature vector of the first layer of the network
    '''
    num_samples = data.shape[0]
    bias = np.dot(np.ones((num_samples,1)),sparse.b1.T)
    z = np.dot(sparse.teth1a, data.T).T + bias
    a = sigmoid(z)
    return a;


