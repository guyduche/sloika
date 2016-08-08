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
                              NonNegative, ParseToNamedTuple, Positive,
                              proportion, Maybe)

from sloika import batch, networks, updates, __version__

# This is here, not in main to allow documentation to be built
parser = argparse.ArgumentParser(
    description='Train Nanonet neural network',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--bad', default=True, action=AutoBool, help='Label bad emissions')
parser.add_argument('--batch', default=300, metavar='size', type=Positive(int),
    help='Batch size (number of chunks to run in parallel)')
parser.add_argument('--adam', nargs=3, metavar=('rate', 'decay1', 'decay2'),
    default=(1e-3, 0.9, 0.999), type=(NonNegative(float), NonNegative(float), NonNegative(float)),
    action=ParseToNamedTuple, help='Parameters for Exponential Decay Adaptive Momementum')
parser.add_argument('--kmer', default=5, metavar='length', type=Positive(int),
    help='Length of kmer to estimate')
parser.add_argument('--limit', default=None, type=Maybe(Positive(int)),
    help='Limit number of reads to process.')
parser.add_argument('--lrdecay', default=None, metavar='epochs', type=Positive(float),
    help='Number of epochs over which learning rate is halved')
parser.add_argument('--model', metavar='file', action=FileExists,
    help='File to read model from')
parser.add_argument('--niteration', metavar='epochs', type=Positive(int), default=1000,
    help='Maximum number of epochs to train for')
parser.add_argument('--save_every', metavar='x', type=Positive(int), default=200,
    help='Save model every x epochs')
parser.add_argument('--sd', default=0.5, metavar='value', type=Positive(float),
    help='Standard deviation to initialise with')
parser.add_argument('--section', default='template', choices=['template', 'complement'],
    help='Section to call')
parser.add_argument('--strand_list', default=None, action=FileExists,
    help='strand summary file containing subset.')
parser.add_argument('--trim', default=(500, 50), nargs=2, type=Positive(int),
    metavar=('beginning', 'end'), help='Number of events to trim off start and end')
parser.add_argument('--use_scaled', default=False, action=AutoBool,
    help='Train from scaled event statistics')
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
        network = networks.nanonet(kmer=args.kmer, winlen=args.window, sd=args.sd, bad_state=args.bad)
    fg, fv = wrap_network(network)

    score = wscore = 0.0
    acc = wacc = 0.0
    SMOOTH = 0.9
    total_ev = 0
    learning_rate = args.adam.rate
    learning_factor = 0.5 ** (1.0 / args.lrdecay) if args.lrdecay is not None else 1.0

    with h5py.File(args.input, 'r') as h5:
        for it in xrange(args.niteration):
            #  Training
            idx = np.sort(np.random.choice(len(h5['chunks']), size=args.batch, replace=False))
            events = np.ascontiguousarray(h5['chunks'][idx, :, :].transpose((1, 0 ,2)))
            labels = np.ascontiguousarray(h5['labels'][idx, :].transpose())

            fval, ncorr = fg(events, labels, learning_rate)
            fval = float(fval)
            ncorr = float(ncorr)
            nev = np.size(labels)
            total_ev += nev
            score = fval + SMOOTH * score
            acc = (ncorr / nev) + SMOOTH * acc
            wscore = 1.0 + SMOOTH * wscore
            wacc = 1.0 + SMOOTH * wacc
            sys.stdout.write('.')
            if (it + 1) % 50 == 0:
                print ' {:5.3f} {:5.2f}%'.format(score / wscore, 100.0 * acc / wacc)

            # Save model
            if (it + 1) % args.save_every == 0:
                with open(args.output + '_epoch{:05d}.pkl'.format(it), 'wb') as fh:
                    cPickle.dump(network, fh, protocol=cPickle.HIGHEST_PROTOCOL)

            learning_rate *= learning_factor

    with open(args.output + '_final.pkl', 'wb') as fh:
        cPickle.dump(network, fh, protocol=cPickle.HIGHEST_PROTOCOL)
