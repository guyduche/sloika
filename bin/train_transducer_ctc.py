#!/usr/bin/env python
import argparse
import cPickle
import numpy as np
import sys
import time

import theano as th
import theano.tensor as T

from ctc import cpu_ctc_th

from untangled import bio, fast5
from untangled.cmdargs import (display_version_and_exit, FileExists
                              NonNegative, ParseToNamedTuple, Positive,
                              proportion, Maybe)

from sloika import networks, updates, features, __version__

# This is here, not in main to allow documentation to be built
parser = argparse.ArgumentParser(
    description='Train a simple transducer neural network',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch', default=1000, metavar='size', type=Positive(int),
    help='Batch size (number of chunks to run in parallel)')
parser.add_argument('--chunk', default=100, metavar='events', type=Positive(int),
    help='Length of each read chunk')
parser.add_argument('--edam', nargs=3, metavar=('rate', 'decay1', 'decay2'),
    default=(0.1, 0.9, 0.99), type=(NonNegative(float), NonNegative(float), NonNegative(float)),
    action=ParseToNamedTuple, help='Parameters for Exponential Decay Adaptive Momementum')
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
parser.add_argument('--sd', default=0.1, metavar='value', type=Positive(float),
    help='Standard deviation to initialise with')
parser.add_argument('--section', default='template', choices=['template', 'complement'],
    help='Section to call')
parser.add_argument('--size', default=64, type=Positive(int), metavar='n',
    help='Hidden layers of network to have size n')
parser.add_argument('--strand_list', default=None, action=FileExists,
    help='strand summary file containing subset.')
parser.add_argument('--trim', default=(500, 50), nargs=2, type=Positive(int),
    metavar=('beginning', 'end'), help='Number of events to trim off start and end')
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
    labels = T.ivector()
    rate = T.scalar()
    post = network.run(x)
    post_len = T.ivector()
    label_len = T.ivector()
    #loss = T.mean(th.map(T.nnet.categorical_crossentropy, sequences=[post, labels])[0])
    loss = T.mean(cpu_ctc_th(post, post_len, labels, label_len))
    update_dict = updates.edam(network, loss, rate, (args.edam.decay1, args.edam.decay2))
    # update_dict = updates.sgd(network, loss, rate, args.edam.decay1)

    fg = th.function([x, post_len, labels, label_len, rate], loss, updates=update_dict)
    fv = th.function([x, post_len, labels, label_len], loss)
    return fg, fv

def chunk_events_ctc(files, max_len, permute=True):
    klen = 1
    kmer_to_state = bio.kmer_mapping(klen)
    black_list = set()

    pfiles = list(files)
    if permute:
        pfiles = np.random.permutation(pfiles)

    in_mat = labels = label_len = None
    for fn in pfiles:
        try:
            with fast5.Reader(fn) as f5:
                ev, _ = f5.get_any_mapping_data(args.section)
        except:
            black_list.add(fn)
            sys.stderr.write('Failed to read from {}.\n'.format(fn))
            continue
        if len(ev) <= sum(args.trim) + args.chunk:
            continue
        ev = ev[args.trim[0] : args.trim[1]]

        new_inMat = features.from_events(ev)
        ml = len(new_inMat) // args.chunk
        new_inMat = new_inMat[:ml * args.chunk].reshape((ml, args.chunk, -1))

        #  Construct label sequence for each row of the event matrix
        new_labels = np.array([], dtype=np.int32)
        new_label_len = np.zeros(len(new_inMat), dtype=np.int32)
        kh = len(ev['kmer'][0]) // 2
        wh = args.window // 2
        valid_labels = np.ones(len(new_inMat), dtype=np.bool)
        for i in xrange(len(new_inMat)):
            offset = i * args.chunk + (args.window // 2)
            kmers = ev['kmer'][offset : offset + args.chunk]
            moves = np.abs(np.ediff1d(ev['seq_pos'][offset : offset + args.chunk]))
            seq = bio.reduce_kmers(kmers, moves)

            states = 1 + np.array(map(lambda k: kmer_to_state[k], bio.seq_to_kmers(seq, klen)), dtype=np.int32)
            new_labels = np.concatenate((new_labels, states))
            new_label_len[i] = len(states)

        new_inMat = new_inMat[valid_labels]
        new_label_len = new_label_len[valid_labels]

        in_mat = np.vstack((in_mat, new_inMat)) if in_mat is not None else new_inMat
        labels = np.concatenate((labels, new_labels)) if labels is not None else new_labels
        label_len = np.concatenate((label_len, new_label_len)) if label_len is not None else new_label_len

        while len(in_mat) > max_len:
            sumlab = np.sum(label_len[:max_len])
            yield (np.ascontiguousarray(in_mat[:max_len].transpose((1,0,2))),
                   np.ascontiguousarray(labels[:sumlab]),
                   np.ascontiguousarray(label_len[:max_len]))
            in_mat = in_mat[max_len:]
            labels = labels[sumlab:]
            label_len = label_len[max_len:]

    if in_mat is not None:
        yield (np.ascontiguousarray(in_mat.transpose((1,0,2))),
               np.ascontiguousarray(labels),
               np.ascontiguousarray(label_len))

    files -= black_list


if __name__ == '__main__':
    args = parser.parse_args()
    kmers = bio.all_kmers(1)

    print '* Creating model'
    if args.model is not None:
        with open(args.model, 'r') as fh:
            network = cPickle.load(fh)
    else:
        network = networks.transducer(winlen=args.window, sd=args.sd, bad_state=False, size=args.size)
    fg, fv = wrap_network(network)

    print '* Reading files'

    train_files = set(fast5.iterate_fast5(args.input_folder, paths=True, limit=args.limit, strand_list=args.strand_list))
    if args.validation is not None:
        nval = 1 + int(args.validation * len(train_files))
        val_files = set(np.random.choice(list(train_files), size=nval, replace=False))
        train_files -= val_files

    print '* Running'
    wh = args.window // 2

    score = wscore = 0.0
    acc = wacc = 0.0
    SMOOTH = 0.95
    learning_rate = args.edam.rate
    learning_factor = 0.5 ** (1.0 / args.lrdecay) if args.lrdecay is not None else 1.0
    for it in xrange(1, args.niteration):
        print '* Epoch {}: learning rate {:6.2e}'.format(it, learning_rate)
        #  Training
        total_ev = 0
        dt = 0.0
        for i, in_data in enumerate(chunk_events_ctc(train_files, args.batch)):
            t0 = time.time()
            lens = np.repeat(in_data[0].shape[0] - 2 * wh, in_data[0].shape[1]).astype(np.int32)
            fval = float(fg(in_data[0], lens, in_data[1], in_data[2], learning_rate)) * 100.0 / args.chunk
            print i, fval
            if i > 10:
                exit(0)
            if fval == 0.0:
                continue

            nev = in_data[0].shape[0] * in_data[0].shape[1]
            total_ev += nev
            score = fval + SMOOTH * score
            wscore = 1.0 + SMOOTH * wscore
            dt += time.time() - t0
            sys.stdout.write('.')
        sys.stdout.write('\n')
        print '  training   {:5.3f} ... {:6.1f}s ({:.2f} kev/s)'.format(score / wscore, dt, 0.001 * total_ev / dt)

        #  Validation
        if args.validation is not None:
            dt = 0.0
            vscore = vnev = vncorr = 0
            for i, in_data in enumerate(chunk_events_ctc(val_files, args.batch)):
                t0 = time.time()
                lens = np.repeat(in_data[0].shape[0] - 2 * wh, in_data[0].shape[1]).astype(np.int32)
                fval = float(fv(in_data[0], lens, in_data[1], in_data[2])) * 100.0 / args.chunk
                nev = in_data[0].shape[0] * in_data[0].shape[1]
                vscore += fval * nev
                vnev += nev
                dt += time.time() - t0
            print '  validation {:5.3f} ... {:6.1f}s ({:.2f} kev/s)'.format(vscore / vnev, dt, 0.001 * vnev / dt)

        # Save model
        if (it % args.save_every) == 0:
            with open(args.output + '_epoch{:05d}.pkl'.format(it), 'wb') as fh:
                cPickle.dump(network, fh, protocol=cPickle.HIGHEST_PROTOCOL)

        learning_rate *= learning_factor

    with open(args.output + '_final.pkl', 'wb') as fh:
        cPickle.dump(network, fh, protocol=cPickle.HIGHEST_PROTOCOL)
