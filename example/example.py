#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import numpy as np
from scipy.io import loadmat
import sys
import shelve
sys.path.append('../')
from encoder import Sparse
import encoder
import visualize

if __name__ == '__main__':

    # create new sparse enconder
    s = Sparse(nodes = 25)
    # load data
    data_mat = loadmat('patch')
    d = data_mat['patches']
    # train sparse encoder
    encoder.fit(s, d, epochs = 600)
    filename = 'persistence.data'
    f = shelve.open(filename)
    f['autoencoder'] = s
    f.close()
    visualize.visualize(s.theta1)
