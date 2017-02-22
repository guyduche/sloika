#!/usr/bin/env python
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import range
from builtins import *
import argparse
import numpy as np
from scipy.linalg import blas
import time

parser = argparse.ArgumentParser()
parser.add_argument('--n1', default=1000, type=int,
                    help='Length of sequence')
parser.add_argument('--n2', default=64, type=int,
                    help='Size of hidden layers')
parser.add_argument('--nst', default=1025, type=int,
                    help='Number of output states')
parser.add_argument('--ntimes', default=100, type=int,
                    help='Number of times to run')
parser.add_argument('--float', default=False, action='store_true',
                    help='Use single precision')
parser.add_argument('--double', dest='float', action='store_false',
                    help='Use double precision')

args = parser.parse_args()
dtype = np.float32 if args.float else np.float64

m = np.empty((args.n1, args.n2), dtype=dtype)
w1 = np.empty((args.n2, args.n2), dtype=dtype)
w2 = np.empty((args.n2, args.nst), dtype=dtype)
w2t = np.ascontiguousarray(w2.transpose())


def ff_softmax(m):
    # 4 Feed forward layers
    np.tanh(np.dot(m, w1), out=m)
    np.tanh(np.dot(m, w1), out=m)
    np.tanh(np.dot(m, w1), out=m)
    np.tanh(np.dot(m, w1), out=m)
    #  Output layer
    a = np.dot(m, w2)
    #  Softmax
    np.exp(a, out=a)
    norm = np.sum(a, axis=1).reshape((-1, 1))
    a /= norm
    return a


def ff2_softmax(m):
    # 4 Feed forward layers
    np.tanh(np.tensordot(m, w1, axes=(1, 1)), out=m)
    np.tanh(np.tensordot(m, w1, axes=(1, 1)), out=m)
    np.tanh(np.tensordot(m, w1, axes=(1, 1)), out=m)
    np.tanh(np.tensordot(m, w1, axes=(1, 1)), out=m)
    #  Output layer
    a = np.tensordot(m, w2t, axes=(1, 1))
    #  Softmax
    np.exp(a, out=a)
    norm = np.sum(a, axis=1).reshape((-1, 1))
    a /= norm
    return a


def ff3_softmax(m):
    matmult = blas.sgemm if args.float else blas.dgemm
    # 4 Feed forward layers
    np.tanh(matmult(0.0, m, w1, trans_b=1), out=m)
    np.tanh(matmult(0.0, m, w1, trans_b=1), out=m)
    np.tanh(matmult(0.0, m, w1, trans_b=1), out=m)
    np.tanh(matmult(0.0, m, w1, trans_b=1), out=m)
    #  Output layer
    a = matmult(0.0, m, w2t, trans_b=1)
    #  Softmax
    np.exp(a, out=a)
    norm = np.sum(a, axis=1).reshape((-1, 1))
    a /= norm
    return a


def ff4_softmax(m):
    # 4 Feed forward layers
    wt = w1.transpose()
    np.tanh(np.dot(m, wt), out=m)
    np.tanh(np.dot(m, wt), out=m)
    np.tanh(np.dot(m, wt), out=m)
    np.tanh(np.dot(m, wt), out=m)
    #  Output layer
    a = np.dot(m, w2.transpose())
    #  Softmax
    np.exp(a, out=a)
    norm = np.sum(a, axis=1).reshape((-1, 1))
    a /= norm
    return a


def timef(f, ntimes):
    t0 = time.time()
    for i in range(ntimes):
        out = f(m)
    return time.time() - t0

print('** Naive implementation')
dt = timef(ff_softmax, args.ntimes)
print('Time = {} ms'.format(dt * 1000.0))
print('Rate = {} kev/s'.format((args.ntimes * args.n1) / dt / 1000.0))

print('** Using tensor dot to work with transpose of matrix')
dt = timef(ff2_softmax, args.ntimes)
print('Time = {} ms'.format(dt * 1000.0))
print('Rate = {} kev/s'.format((args.ntimes * args.n1) / dt / 1000.0))

print('** Using BLAS directly from scipy')
dt = timef(ff3_softmax, args.ntimes)
print('Time = {} ms'.format(dt * 1000.0))
print('Rate = {} kev/s'.format((args.ntimes * args.n1) / dt / 1000.0))

print('** Numpy transpose')
dt = timef(ff_softmax, args.ntimes)
print('Time = {} ms'.format(dt * 1000.0))
print('Rate = {} kev/s'.format((args.ntimes * args.n1) / dt / 1000.0))
