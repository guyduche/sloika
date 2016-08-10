#!/usr/bin/env python
import argparse
import cPickle
import h5py
import numpy as np
import sys
import time

import theano as th
import theano.tensor as T

from ctc import cpu_ctc_th

from untangled import bio, fast5
from untangled.cmdargs import (display_version_and_exit, FileExists,
                              NonNegative, ParseToNamedTuple, Positive,
                              proportion, Maybe)

from sloika import networks, updates, features, __version__

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
    labels = T.ivector()
    rate = T.scalar()
    post = network.run(x)
    post_len = T.ivector()
    label_len = T.ivector()
    #loss = T.mean(th.map(T.nnet.categorical_crossentropy, sequences=[post, labels])[0])
    loss = T.mean(cpu_ctc_th(post, post_len, labels, label_len))
    update_dict = updates.adam(network, loss, rate, (args.adam.decay1, args.adam.decay2))
    # update_dict = updates.sgd(network, loss, rate, args.adam.decay1)

    fg = th.function([x, post_len, labels, label_len, rate], loss, updates=update_dict)
    fv = th.function([x, post_len, labels, label_len], loss)
    return fg, fv


def compress_labels(labels, klen):
    """ Make a CTC compatible vector of labels

    :param labels: A :class:`ndarray` (batch x chunk) containing labels

    :returns: A tuple of a :class:`ndarray` containing the concatenated
    references sequences and a :class:`ndarrray` containing the reference
    lengths
    """
    state_to_kmer = bio.all_kmers(klen)
    kmer_to_state = bio.kmer_mapping(klen)
    label_vec = np.empty(0, dtype=np.int32)
    label_len = np.empty(len(labels), dtype=np.int32)
    for i, lbl in enumerate(labels):
        label_kmers = [state_to_kmer[s - 1] for s in lbl if s > 0]
        seq = bio.kmers_to_sequence(label_kmers, homopolymer_step=True)
        seq_kmers = bio.seq_to_kmers(seq, klen)
        label_vec = 1 + np.append(label_vec, [kmer_to_state[k] for k in seq_kmers])
        label_len[i] = len(seq_kmers)

    return label_vec.astype(np.int32), label_len


if __name__ == '__main__':
    args = parser.parse_args()
    kmers = bio.all_kmers(args.kmer)

    print '* Creating model'
    if args.model is not None:
        with open(args.model, 'r') as fh:
            network = cPickle.load(fh)
    else:
        network = networks.transducer(winlen=args.window, sd=args.sd,
                                      size=args.size, klen=args.kmer)
    fg, fv = wrap_network(network)

    with h5py.File(args.input, 'r') as h5:
        all_chunks = h5['chunks'][:]
        all_labels = h5['labels'][:]
        assert not np.any(all_labels[:, 0] == 0)

    total_ev = 0
    score = wscore = 0.0
    SMOOTH = 0.99
    learning_rate = args.adam.rate
    learning_factor = 0.5 ** (1.0 / args.lrdecay) if args.lrdecay is not None else 1.0

    t0 = time.time()
    for i in xrange(args.niteration):
        idx = np.sort(np.random.choice(len(all_chunks), size=args.batch, replace=False))
        events = np.ascontiguousarray(all_chunks[idx].transpose((1, 0, 2)))
        label_vec, label_lens = compress_labels(all_labels[idx], args.kmer)
        chunk_size = events.shape[0]

        lens = np.repeat(events.shape[0] - args.window + 1, events.shape[1]).astype(np.int32)
        fval = float(fg(events, lens, label_vec, label_lens, learning_rate))
        fval *= 100.0 / chunk_size

        nev = chunk_size * args.batch
        total_ev += nev
        score = fval + SMOOTH * score
        wscore = 1.0 + SMOOTH * wscore
        sys.stdout.write('.')
        if (i + 1) % 50 == 0:
            tn = time.time()
            dt = tn - t0
            print ' {:5d} {:5.3f}  {:5.2f}%  {:5.2f}s ({:.2f} kev/s)'.format((i + 1) // 50, score / wscore, dt, total_ev / 1000.0 / dt)
            total_ev = 0
            t0 = tn

        # Save model
        if (i + 1) % args.save_every == 0:
            with open(args.output + '_{:05d}.pkl'.format((i + 1) // args.save_every), 'wb') as fh:
                cPickle.dump(network, fh, protocol=cPickle.HIGHEST_PROTOCOL)

        learning_rate *= learning_factor

    with open(args.output + '_final.pkl', 'wb') as fh:
        cPickle.dump(network, fh, protocol=cPickle.HIGHEST_PROTOCOL)
