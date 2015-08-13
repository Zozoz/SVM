#!/usr/bin/env python
# encoding: utf-8


import numpy as np


def GaussianKernel(X, Z, sigma):
    '''
        Gaussian Kernel function.
        X, Z is vector column (n, 1).
        simga is standard deviation.
    '''
    K = np.exp(-np.dot(X.T, Z) / (2 * sigma**2))
    return K


class SVM(object):

    def __init__(self, data_x, data_y, C=0.06):
        self.data_x = data_x
        self.data_y = data_y
        self.C = C
        self.m, self.n = np.shape(self.data_x)
        self.K = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):
            for j in range(self.m):
                self.K[i, j] = GaussianKernel(self.data_x[i, :], self.data_x[j, :], 0)
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.support_vector = None
        self.support_vector_alphas = None
        self.support_vector_y = None

    def get_gx(self, data_x, data_index):
        gx = self.alphas.T * np.multiply(self.data_y, self.K[data_index, :]) + self.b
        return gx
