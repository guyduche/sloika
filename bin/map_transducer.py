#!/usr/bin/env python
import argparse
import h5py
from six.moves import cPickle
import sys
import time

from dragonet.bio import seq_tools

import numpy as np
import numpy.lib.recfunctions as nprf
from tang.fast5 import iterate_fast5, fast5
from sloika import features, transducer
from tang.util.cmdargs import (AutoBool, display_version_and_exit, FileExist,
                               NonNegative, probability, Positive, TypeOrNone,
                               Vector)
from tang.util.tang_iter import tang_imap

# This is here, not in main to allow documentation to be built
parser = argparse.ArgumentParser(
    description='Map transducer to reference sequence',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--limit', default=None, type=TypeOrNone(Positive(int)),
    help='Limit number of reads to process.')
parser.add_argument('--section', default='template', choices=['template', 'complement'],
    help='Section to call')
parser.add_argument('--slip', default=None, type=TypeOrNone(NonNegative(float)),
    help='Slip penalty')
parser.add_argument('--strand_list', default=None, action=FileExist,
    help='strand summary file containing subset.')
parser.add_argument('--trim', default=(500, 50), nargs=2, type=Positive(int),
    metavar=('beginning', 'end'), help='Number of events to trim off start and end')
parser.add_argument('--version', nargs=0, action=display_version_and_exit,
    help='Display version information.')
parser.add_argument('--window', default=3, type=Positive(int), metavar='length',
    help='Window length for input features')
parser.add_argument('model', action=FileExist, help='Pickled model file')
parser.add_argument('output', help='HDF5 file for output')
parser.add_argument('input_folder', action=FileExist,
    help='Directory containing single-read fast5 files.')

def map_transducer(args, fn):
    _, kmer_to_state = seq_tools.all_kmers(length=1, rev_map=True)
    try:
        with fast5(fn) as f5:
            ev, _ = f5.get_any_mapping_data(args.section)
            name, seq = f5.get_reference_fasta(section=args.section).split()
            sn = f5.filename_short
    except:
        return None
    if len(ev) <= sum(args.trim):
        return None

    inMat = features.from_events(ev)[args.trim[0] : -args.trim[1]]
    inMat = np.expand_dims(inMat, axis=1)

    with open(args.model, 'r') as fh:
        calc_post = cPickle.load(fh)

    trans = np.squeeze(calc_post(inMat))
    seq = np.array(map(lambda k: kmer_to_state[k], seq))
    score, path = transducer.map_to_sequence(trans, seq, slip=args.slip, log=False)
    mp_rnn = np.argmax(trans, axis=1)

    lb = args.trim[0] + 1
    ub = args.trim[1] + 1
    ev = ev[lb:-ub]
    lbls = np.array(map(lambda k: kmer_to_state[k[2]], ev['kmer']))
    lbls[np.ediff1d(ev['seq_pos'], to_begin=1) == 0] = 4

    return sn, score, path, ev, lbls, mp_rnn, seq

if __name__ == '__main__':
    args = parser.parse_args()

    files = iterate_fast5(args.input_folder, paths=True, limit=args.limit, strand_list=args.strand_list)
    nbases = nevents = 0
    t0 = time.time()
    print 'Read\tOldAcc\tNewAcc\tDelta'
    with h5py.File(args.output, 'w') as h5:
        for res in tang_imap(map_transducer, files, threads=1, fix_args=[args], unordered=True):
            if res is None:
                continue
            read, score, path, ev, lbls, rnn_mp, seq = res
            rnn_call = seq[path]
            rnn_call[np.ediff1d(path, to_begin=1) == 0] = 4
            acc_old = np.mean(lbls == rnn_mp)
            acc_new = np.mean(rnn_call == rnn_mp)
            acc_delta = acc_new - acc_old
            ev = nprf.append_fields(ev, ('rnn_pos', 'rnn_mp', 'rnn_call'), (path, rnn_mp.astype(np.int32), rnn_call.astype(np.int32)))
            h5[read] = ev
            print read, '{:5.2f}\t{:5.2f}\t{:5.2f}'.format(
                100 * acc_old, 100 * acc_new, 100 * acc_delta)
    dt = time.time() - t0
