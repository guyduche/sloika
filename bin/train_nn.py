#!/usr/bin/env python
import argparse
import cPickle
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
parser.add_argument('--chunk', default=500, metavar='events', type=Positive(int),
    help='Length of each read chunk')
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
parser.add_argument('--niteration', metavar='epochs', type=Positive(int), default=500,
    help='Maximum number of epochs to train for')
parser.add_argument('--save_every', metavar='x', type=Positive(int), default=5,
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
parser.add_argument('--validation', default=None, type=proportion,
    help='Proportion of reads to use for validation')
parser.add_argument('--version', nargs=0, action=display_version_and_exit, metavar=__version__,
    help='Display version information.')
parser.add_argument('--window', default=3, type=Positive(int), metavar='length',
    help='Window length for input features')
parser.add_argument('output', help='Prefix for output files')
parser.add_argument('input_folder', action=FileExists,
    help='Directory containing single-read fast5 files.')

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

    train_files = set(fast5.iterate_fast5(args.input_folder, paths=True, limit=args.limit, strand_list=args.strand_list))
    if args.validation is not None:
        nval = 1 + int(args.validation * len(train_files))
        val_files = set(np.random.choice(list(train_files), size=nval, replace=False))
        train_files -= val_files

    score = wscore = 0.0
    acc = wacc = 0.0
    SMOOTH = 0.9
    learning_rate = args.adam.rate
    learning_factor = 0.5 ** (1.0 / args.lrdecay) if args.lrdecay is not None else 1.0
    for it in xrange(args.niteration):
        print '* Epoch {}: learning rate {:6.2e}'.format(it + 1, learning_rate)
        #  Training
        total_ev = 0
        t0 = time.time()
        for i, in_data in enumerate(batch.kmers(train_files, args.section,
                                                args.batch, args.chunk,
                                                args.window, args.kmer,
                                                trim=args.trim, bad=args.bad,
                                                use_scaled=args.use_scaled)):
            fval, ncorr = fg(in_data[0], in_data[1], learning_rate)
            fval = float(fval)
            ncorr = float(ncorr)
            nev = np.size(in_data[1])
            total_ev += nev
            score = fval + SMOOTH * score
            acc = (ncorr / nev) + SMOOTH * acc
            wscore = 1.0 + SMOOTH * wscore
            wacc = 1.0 + SMOOTH * wacc
            sys.stdout.write('.')
            if (i + 1) % 50 == 0:
                print "{:8d} : {:8.4f} {:8.4f}".format(i + 1, fval, score / wscore)
        sys.stdout.write('\n')
        dt = time.time() - t0
        print '  training   {:5.3f}   {:5.2f}% ... {:6.1f}s ({:.2f} kev/s)'.format(score / wscore, 100.0 * acc / wacc, dt, 0.001 * total_ev / dt)

        #  Validation
        if args.validation is not None:
            t0 = time.time()
            vscore = vnev = vncorr = 0
            for i, in_data in enumerate(batch.kmers(val_files, args.section,
                                                    args.batch, args.chunk,
                                                    args.window, args.kmer,
                                                    trim=args.trim, bad=args.bad,
                                                    use_scaled=args.use_scaled)):
                fval, ncorr = fv(in_data[0], in_data[1])
                fval = float(fval)
                ncorr = float(ncorr)
                nev = np.size(in_data[1])
                vscore += fval * nev
                vncorr += ncorr
                vnev += nev
                sys.stdout.write('.')
                if (i + 1) % 50 == 0:
                    print "{:8d} : {:8.4f} {:8.4f}".format(i + 1, fval, vscore / vnev)
            sys.stdout.write('\n')
            dt = time.time() - t0
            print '  validation {:5.3f}   {:5.2f}% ... {:6.1f}s ({:.2f} kev/s)'.format(vscore / vnev, 100.0 * vncorr / vnev, dt, 0.001 * vnev / dt)

        # Save model
        if (it % args.save_every) == 0:
            with open(args.output + '_epoch{:05d}.pkl'.format(it), 'wb') as fh:
                cPickle.dump(network, fh, protocol=cPickle.HIGHEST_PROTOCOL)

        learning_rate *= learning_factor

    with open(args.output + '_final.pkl', 'wb') as fh:
        cPickle.dump(network, fh, protocol=cPickle.HIGHEST_PROTOCOL)
