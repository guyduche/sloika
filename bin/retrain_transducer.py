#!/usr/bin/env python
import argparse
import cPickle
import h5py
import numpy as np
import sys
import time
import warnings

import theano as th
import theano.tensor as T

from untangled import bio
from untangled.cmdargs import (AutoBool, display_version_and_exit, FileExist,
                              NonNegative, ParseToNamedTuple, Positive,
                              probability)

from sloika import networks, updates, features, sloika_dtype, __version__

# This is here, not in main to allow documentation to be built
parser = argparse.ArgumentParser(
    description='Retrain a simple transducer neural network from output of map_transducer.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch', default=1000, metavar='size', type=Positive(int),
    help='Batch size (number of chunks to run in parallel)')
parser.add_argument('--chunk', default=100, metavar='events', type=Positive(int),
    help='Length of each read chunk')
parser.add_argument('--drop_runs', metavar='length', type=Positive(int), default=10,
    help='Drop chunks with runs longer than length')
parser.add_argument('--edam', nargs=3, metavar=('rate', 'decay1', 'decay2'),
    default=(0.1, 0.9, 0.99), type=(NonNegative(float), NonNegative(float), NonNegative(float)),
    action=ParseToNamedTuple, help='Parameters for Exponential Decay Adaptive Momementum')
parser.add_argument('--lrdecay', default=None, metavar='epochs', type=Positive(float),
    help='Number of epochs over which learning rate is halved')
parser.add_argument('--model', metavar='file', action=FileExist,
    help='File to read model from')
parser.add_argument('--niteration', metavar='epochs', type=Positive(int), default=500,
    help='Maximum number of epochs to train for')
parser.add_argument('--save_every', metavar='x', type=Positive(int), default=5,
    help='Save model every x epochs')
parser.add_argument('--sd', default=0.1, metavar='value', type=Positive(float),
    help='Standard deviation to initialise with')
parser.add_argument('--size', default=64, type=Positive(int), metavar='n',
    help='Hidden layers of network to have size n')
parser.add_argument('--validation', default=None, type=probability,
    help='Proportion of reads to use for validation')
parser.add_argument('--version', nargs=0, action=display_version_and_exit, metavar=__version__,
    help='Display version information.')
parser.add_argument('--window', default=3, type=Positive(int), metavar='length',
    help='Window length for input features')
parser.add_argument('output', help='Prefix for output files')
parser.add_argument('input', action=FileExist, help='HDF5 file.')

_ETA = 1e-300
_NBASE = 4


def wrap_network(network):
    x = T.tensor3(dtype=sloika_dtype)
    labels = T.imatrix()
    rate = T.scalar(dtype=sloika_dtype)
    post = network.run(x)
    loss = T.mean(th.map(T.nnet.categorical_crossentropy, sequences=[post, labels])[0])
    ncorrect = T.sum(T.eq(T.argmax(post,  axis=2), labels))
    update_dict = updates.edam(network, loss, rate, (args.edam.decay1, args.edam.decay2))
    # update_dict = updates.sgd(network, loss, rate, args.edam.decay1)

    fg = th.function([x, labels, rate], [loss, ncorrect], updates=update_dict)
    fv = th.function([x, labels], [loss, ncorrect])
    return fg, fv


def chunk_events(infile, files, max_len, permute=True):

    with h5py.File(infile, 'r') as h5:
        pfiles = list(files & set(h5.keys()))
    if permute:
        pfiles = np.random.permutation(pfiles)

    in_mat = labels = None
    for fn in pfiles:
        with h5py.File(infile, 'r') as h5:
            ev = h5[fn][:]
        if len(ev) <= args.chunk:
            continue

        new_inMat = features.from_events(ev).astype(np.float32)
        ml = len(new_inMat) // args.chunk
        new_inMat = new_inMat[:ml * args.chunk].reshape((ml, args.chunk, -1))

        ub = args.chunk * ml
        new_labels = ev['rnn_call'][:ub].astype(np.int32)
        new_labels = new_labels.reshape((ml, args.chunk))
        new_labels = new_labels[:, (args.window // 2) : -(args.window // 2)]

        in_mat = np.vstack((in_mat, new_inMat)) if in_mat is not None else new_inMat
        labels = np.vstack((labels, new_labels)) if labels is not None else new_labels
        while len(in_mat) > max_len:
            yield (np.ascontiguousarray(in_mat[:max_len].transpose((1,0,2))),
                   np.ascontiguousarray(labels[:max_len].transpose()))
            in_mat = in_mat[max_len:]
            labels = labels[max_len:]



if __name__ == '__main__':
    warnings.simplefilter("always", DeprecationWarning)

    args = parser.parse_args()
    kmers = bio.all_kmers(1)

    if args.model is not None:
        with open(args.model, 'r') as fh:
            network = cPickle.load(fh)
    else:
        network = networks.transducer(winlen=args.window, sd=args.sd, bad_state=False, size=args.size)
    fg, fv = wrap_network(network)

    with h5py.File(args.input, 'r') as f5:
        train_files = set(f5.keys())
    if args.validation is not None:
        nval = 1 + int(args.validation * len(train_files))
        val_files = set(np.random.choice(list(train_files), size=nval, replace=False))
        train_files -= val_files

    score = wscore = 0.0
    acc = wacc = 0.0
    SMOOTH = 0.8
    learning_rate = args.edam.rate
    learning_factor = 0.5 ** (1.0 / args.lrdecay) if args.lrdecay is not None else 1.0
    for it in xrange(1, args.niteration):
        print '* Epoch {}: learning rate {:6.2e}'.format(it, learning_rate)
        #  Training
        total_ev = 0
        dt = 0.0
        for i, in_data in enumerate(chunk_events(args.input, train_files, args.batch)):
            sys.stdout.write('.')
            t0 = time.time()
            fval, ncorr = fg(in_data[0], in_data[1], learning_rate)
            fval = float(fval)
            ncorr = float(ncorr)
            nev = in_data[1].shape[0] * in_data[1].shape[1]
            total_ev += nev
            score = fval + SMOOTH * score
            acc = (ncorr / nev) + SMOOTH * acc
            wscore = 1.0 + SMOOTH * wscore
            wacc = 1.0 + SMOOTH * wacc
            dt += time.time() - t0
            print i, ncorr / nev
        sys.stdout.write('\n')
        print '  training   {:5.3f}   {:5.2f}% ... {:6.1f}s ({:.2f} kev/s)'.format(score / wscore, 100.0 * acc / wacc, dt, 0.001 * total_ev / dt)

        #  Validation
        if args.validation is not None:
            dt = 0.0
            vscore = vnev = vncorr = 0
            for i, in_data in enumerate(chunk_events(args.input, val_files, args.batch)):
                sys.stdout.write('.')
                t0 = time.time()
                fval, ncorr = fv(in_data[0], in_data[1])
                fval = float(fval)
                ncorr = float(ncorr)
                nev = in_data[1].shape[0] * in_data[1].shape[1]
                vscore += fval * nev
                vncorr += ncorr
                vnev += nev
                dt += time.time() - t0
            sys.stdout.write('\n')
            print '  validation {:5.3f}   {:5.2f}% ... {:6.1f}s ({:.2f} kev/s)'.format(vscore / vnev, 100.0 * vncorr / vnev, dt, 0.001 * vnev / dt)

        # Save model
        if (it % args.save_every) == 0:
            with open(args.output + '_epoch{:05d}.pkl'.format(it), 'wb') as fh:
                cPickle.dump(network, fh, protocol=cPickle.HIGHEST_PROTOCOL)

        learning_rate *= learning_factor

    with open(args.output + '_final.pkl', 'wb') as fh:
        cPickle.dump(network, fh, protocol=cPickle.HIGHEST_PROTOCOL)
