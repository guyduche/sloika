#!/usr/bin/env python
import argparse
import cPickle
import h5py
import imp
import numpy as np
import os
from shutil import copyfile
import sys
import time

import theano as th
import theano.tensor as T

from untangled.cmdargs import (AutoBool, display_version_and_exit, FileAbsent,
                               FileExists, Maybe, NonNegative, ParseToNamedTuple,
                               Positive, proportion)

from sloika import updates, __version__

# This is here, not in main to allow documentation to be built
parser = argparse.ArgumentParser(
    description='Train a simple neural network',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--adam', nargs=3, metavar=('rate', 'decay1', 'decay2'),
                    default=(1e-3, 0.9, 0.999), type=(NonNegative(float), NonNegative(float), NonNegative(float)),
                    action=ParseToNamedTuple, help='Parameters for Exponential Decay Adaptive Momementum')
parser.add_argument('--bad', default=True, action=AutoBool,
                    help='Use bad events as a separate state')
parser.add_argument('--batch', default=100, metavar='size', type=Positive(int),
                    help='Batch size (number of chunks to run in parallel)')
parser.add_argument('--drop', default=0, metavar='events', type=NonNegative(int),
                    help='Drop a number of events from start and end of chunk before evaluating loss')
parser.add_argument('--ilf', default=False, action=AutoBool,
                    help='Weight objective function by Inverse Label Frequency')
parser.add_argument('--l2', default=0.0, metavar='penalty', type=NonNegative(float),
                    help='L2 penalty on parameters')
parser.add_argument('--lrdecay', default=5000, metavar='batches', type=Positive(float),
                    help='Number of batches to halving of learning rate')
parser.add_argument('--min_prob', default=0.0, metavar='p', type=proportion,
                    help='Minimum probability allowed for training')
parser.add_argument('--niteration', metavar='batches', type=Positive(int), default=50000,
                    help='Maximum number of batches to train for')
parser.add_argument('--reweight', metavar='group', default='weights', type=Maybe(str),
                    help="Select chunk according to weights in 'group'")
parser.add_argument('--save_every', metavar='x', type=Positive(int), default=5000,
                    help='Save model every x batches')
parser.add_argument('--sd', default=0.5, metavar='value', type=Positive(float),
                    help='Standard deviation to initialise with')
parser.add_argument('--seed', default=None, metavar='integer', type=Positive(int),
                    help='Set random number seed')
parser.add_argument('--transducer', default=True, action=AutoBool,
                    help='Train a transducer based model')
parser.add_argument('--version', nargs=0, action=display_version_and_exit, metavar=__version__,
                    help='Display version information.')
parser.add_argument('model', action=FileExists,
                    help='File to read python model description from')
parser.add_argument('output', action=FileAbsent, help='Prefix for output files')
parser.add_argument('input', action=FileExists,
                    help='HDF5 file containing chunks')


def remove_blanks(labels):
    for lbl_ch in labels:
        for i in xrange(1, len(lbl_ch)):
            if lbl_ch[i] == 0:
                lbl_ch[i] = lbl_ch[i - 1]
    return labels


def wrap_network(network, min_prob=0.0, l2=0.0, drop=0):
    ldrop, udrop = drop, -drop
    if udrop == 0:
        udrop = None

    x = T.tensor3()
    labels = T.imatrix()
    weights = T.fmatrix()
    rate = T.scalar()
    post = min_prob + (1.0 - min_prob) * network.run(x)
    penalty = l2 * updates.param_sqr(network)

    loss_per_event, _ = th.map(T.nnet.categorical_crossentropy, sequences=[post, labels])
    loss = penalty + T.mean((weights * loss_per_event)[ldrop: udrop])
    ncorrect = T.sum(T.eq(T.argmax(post,  axis=2), labels))
    update_dict = updates.adam(network, loss, rate, (args.adam.decay1, args.adam.decay2))

    fg = th.function([x, labels, weights, rate], [loss, ncorrect], updates=update_dict)
    return fg


def saveModel(log, network, output, index):
    log.write('C')
    with open(os.path.join(args.output, 'model_checkpoint_{:05d}.pkl'.format((i + 1) // args.save_every)), 'wb') as fh:
        cPickle.dump(network, fh, protocol=cPickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    args = parser.parse_args()
    np.random.seed(args.seed)

    os.mkdir(args.output)
    copyfile(args.model, os.path.join(args.output, 'model.py'))

    log = open(os.path.join(args.output, 'model.log'), 'w', 0)

    log.write('* Command line\n')
    log.write(' '.join(sys.argv) + '\n')

    log.write('* Reading network from {}\n'.format(args.model))
    model_ext = os.path.splitext(args.model)[1]
    if model_ext == '.py':
        with h5py.File(args.input, 'r') as h5:
            klen = h5.attrs['kmer']
        netmodule = imp.load_source('netmodule', args.model)
        network = netmodule.network(klen=klen, sd=args.sd)
    elif model_ext == '.pkl':
        with open(args.model, 'r') as fh:
            network = cPickle.load(fh)
    else:
        log.write('* Model is neither python file nor model pickle\n')
        exit(1)
    fg = wrap_network(network, min_prob=args.min_prob, l2=args.l2, drop=args.drop)

    log.write('* Loading data from {}\n'.format(args.input))
    with h5py.File(args.input, 'r') as h5:
        all_chunks = h5['chunks'][:]
        all_labels = h5['labels'][:]
        all_bad = h5['bad'][:]
        if args.reweight is not None:
            all_weights = h5[args.reweight][:]
        else:
            all_weights = np.ones(len(all_chunks))

    all_weights /= np.sum(all_weights)

    if not args.transducer:
        remove_blanks(all_labels)

    if args.bad:
        all_labels[all_bad] = 0

    if args.ilf:
        #  Calculate label weights using inverse frequency
        label_weights = np.zeros(np.max(all_labels) + 1, dtype='f4')
        for i, lbls in enumerate(all_labels):
            label_weights += all_weights[i] * np.bincount(lbls, minlength=len(label_weights))
    else:
        label_weights = np.ones(np.max(all_labels) + 1, dtype='f4')
    label_weights = np.reciprocal(label_weights)
    label_weights /= np.mean(label_weights)

    total_ev = 0
    score = wscore = 0.0
    acc = wacc = 0.0
    SMOOTH = 0.8
    lrfactor = 0.0 if args.lrdecay is None else (1.0 / args.lrdecay)

    t0 = time.time()
    log.write('* Training\n')
    saveModel(log, network, args.output, 0)
    for i in xrange(args.niteration):
        learning_rate = args.adam.rate / (1.0 + i * lrfactor)
        idx = np.sort(np.random.choice(len(all_chunks), size=args.batch,
                                       replace=False, p=all_weights))
        events = np.ascontiguousarray(all_chunks[idx].transpose((1, 0, 2)))
        labels = np.ascontiguousarray(all_labels[idx].transpose())
        weights = label_weights[labels]

        fval, ncorr = fg(events, labels, weights, learning_rate)
        fval = float(fval)
        ncorr = float(ncorr)
        nev = np.size(labels)
        total_ev += nev
        score = fval + SMOOTH * score
        acc = (ncorr / nev) + SMOOTH * acc
        wscore = 1.0 + SMOOTH * wscore
        wacc = 1.0 + SMOOTH * wacc

        if (i + 1) % args.save_every == 0:
            saveModel(log, network, args.output, (i + 1) // args.save_every)
        else:
            log.write('.')

        if (i + 1) % 50 == 0:
            tn = time.time()
            dt = tn - t0
            log.write(' {:5d} {:5.3f}  {:5.2f}%  {:5.2f}s ({:.2f} kev/s)\n'.format((i + 1) //
                                                                                   50, score / wscore, 100.0 * acc / wacc, dt, total_ev / 1000.0 / dt))
            total_ev = 0
            t0 = tn

    with open(os.path.join(args.output,  'model_final.pkl'), 'wb') as fh:
        cPickle.dump(network, fh, protocol=cPickle.HIGHEST_PROTOCOL)
