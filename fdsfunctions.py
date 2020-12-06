import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd


def sigmoid(x):
    # return the sigmoid of x
    g = 1 / ( 1 + np.exp(-x))
    return g

def log_likelihood(theta, x, y):
    # return the log likehood of theta according to data x and label y
    m = x.shape[0]
    h = sigmoid(np.dot(x, theta))
    log_l =  ( np.dot(y, np.log(h)) + np.dot((1 - y), np.log(1 - h)) ) / m
    return log_l

def grad_l(theta, x, y):
    # return the gradient G of the log likelihood
    #m = x.shape[0]
    h = sigmoid(np.dot(x, theta))
    G = np.dot(np.transpose(x), y - h)
    return G

def gradient_ascent(theta, x, y, G, alpha=0.01, iterations=100):

    m = len(y)
    log_l_history = np.zeros(iterations)
    theta_history = np.zeros((iterations, x.shape[1]))

    # return the optimized theta parameters,
    # as well as two lists containing the log likelihood's and values of theta at all iterations
    for i in range(iterations):
        theta = theta + alpha * G(theta, x, y) / m
        theta = np.array(theta, dtype=np.float32)
        theta_history[i] = theta
        log_l_history[i] = log_likelihood(theta, x, y)
    return theta, log_l_history, theta_history

def rescale(data: np.array):
    """
    Rescales data into [0, 1] range
    """
    for i in range(data.shape[1]):
        column = data[:, i]
        column = (column - column.min()) / (column.max() - column.min())
        data[:, i] = column
