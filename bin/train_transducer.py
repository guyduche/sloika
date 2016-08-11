#!/usr/bin/env python
import argparse
import cPickle
import h5py
import numpy as np
import sys
import time

import theano as th
import theano.tensor as T

from untangled import bio, fast5
from untangled.cmdargs import (AutoBool, display_version_and_exit, FileExists,
                               Maybe, NonNegative, ParseToNamedTuple, Positive,
                               proportion)

from sloika import networks, updates, __version__

# This is here, not in main to allow documentation to be built
parser = argparse.ArgumentParser(
    description='Train a simple transducer neural network',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch', default=1000, metavar='size', type=Positive(int),
    help='Batch size (number of chunks to run in parallel)')
parser.add_argument('--adam', nargs=3, metavar=('rate', 'decay1', 'decay2'),
    default=(1e-3, 0.9, 0.999), type=(NonNegative(float), NonNegative(float), NonNegative(float)),
    action=ParseToNamedTuple, help='Parameters for Exponential Decay Adaptive Momementum')
parser.add_argument('--kmer', default=1, metavar='length', type=Positive(int),
    help='Length of kmer transducer to train')
parser.add_argument('--lrdecay', default=None, metavar='batches', type=Positive(float),
    help='Number of batches over which learning rate is halved')
parser.add_argument('--model', metavar='file', action=FileExists,
    help='File to read model from')
parser.add_argument('--niteration', metavar='batches', type=Positive(int), default=1000,
    help='Maximum number of batches to train for')
parser.add_argument('--save_every', metavar='x', type=Positive(int), default=200,
    help='Save model every x batches')
parser.add_argument('--sd', default=0.1, metavar='value', type=Positive(float),
    help='Standard deviation to initialise with')
parser.add_argument('--size', default=64, type=Positive(int), metavar='n',
    help='Hidden layers of network to have size n')
parser.add_argument('--validation', default=None, type=proportion,
    help='Proportion of reads to use for validation')
parser.add_argument('--version', nargs=0, action=display_version_and_exit, metavar=__version__,
    help='Display version information.')
parser.add_argument('--window', default=3, type=Positive(int), metavar='length',
    help='Window length for input features')
parser.add_argument('output', help='Prefix for output files')
parser.add_argument('input', action=FileExists,
    help='HDF5 file containing chunks')

_ETA = 1e-300
_NBASE = 4


def wrap_network(network):
    x = T.tensor3()
    labels = T.imatrix()
    rate = T.scalar()
    post = network.run(x)
    loss = T.mean(th.map(T.nnet.categorical_crossentropy, sequences=[post, labels])[0])
    ncorrect = T.sum(T.eq(T.argmax(post,  axis=2), labels))
    update_dict = updates.adam(network, loss, rate, (args.adam.decay1, args.adam.decay2))
    # update_dict = updates.sgd(network, loss, rate, args.adam.decay1)

    fg = th.function([x, labels, rate], [loss, ncorrect], updates=update_dict)
    fv = th.function([x, labels], [loss, ncorrect])
    return fg, fv


if __name__ == '__main__':
    args = parser.parse_args()
    kmers = bio.all_kmers(args.kmer)

    if args.model is not None:
        with open(args.model, 'r') as fh:
            network = cPickle.load(fh)
    else:
        network = networks.transducer(winlen=args.window, size=args.size,
                                      sd=args.sd, klen=args.kmer)
    fg, fv = wrap_network(network)

    with h5py.File(args.input, 'r') as h5:
        full_chunks = h5['chunks'][:]
        full_labels = h5['labels'][:]
    assert not np.any(full_labels[:, 0] == 0)


    total_ev = 0
    score = wscore = 0.0
    acc = wacc = 0.0
    SMOOTH = 0.8
    learning_rate = args.adam.rate
    learning_factor = 0.5 ** (1.0 / args.lrdecay) if args.lrdecay is not None else 1.0

    t0 = time.time()
    for i in xrange(args.niteration):
        idx = np.sort(np.random.choice(len(full_chunks), size=args.batch, replace=False))
        events = np.ascontiguousarray(full_chunks[idx].transpose((1, 0, 2)))
        labels = np.ascontiguousarray(full_labels[idx].transpose())

        fval, ncorr = fg(events, labels, learning_rate)
        fval = float(fval)
        ncorr = float(ncorr)
        nev = np.size(labels)
        total_ev += nev
        score = fval + SMOOTH * score
        acc = (ncorr / nev) + SMOOTH * acc
        wscore = 1.0 + SMOOTH * wscore
        wacc = 1.0 + SMOOTH * wacc

        # Save model
        if (i + 1) % args.save_every == 0:
            sys.stdout.write('C')
            with open(args.output + '_{:05d}.pkl'.format((i + 1) // args.save_every), 'wb') as fh:
                cPickle.dump(network, fh, protocol=cPickle.HIGHEST_PROTOCOL)
        else:
            sys.stdout.write('.')

        if (i + 1) % 50 == 0:
            tn = time.time()
            dt = tn - t0
            print ' {:5d} {:5.3f}  {:5.2f}%  {:5.2f}s ({:.2f} kev/s)'.format((i + 1) // 50, score / wscore, 100.0 * acc / wacc, dt, total_ev / 1000.0 / dt)
            total_ev = 0
            t0 = tn

        learning_rate *= learning_factor

    with open(args.output + '_final.pkl', 'wb') as fh:
        cPickle.dump(network, fh, protocol=cPickle.HIGHEST_PROTOCOL)
