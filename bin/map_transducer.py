#!/usr/bin/env python
import argparse
from six.moves import cPickle
import sys
import time

from dragonet.bio import seq_tools

import numpy as np
import numpy.lib.recfunctions as nprf
import tang.util.io as tangio
from tang.fast5 import iterate_fast5, fast5
from sloika import features, transducer
from tang.util.cmdargs import (AutoBool, display_version_and_exit, FileExist,
                               probability, Positive, TypeOrNone, Vector)
from tang.util.tang_iter import tang_imap

# This is here, not in main to allow documentation to be built
parser = argparse.ArgumentParser(
    description='Map transducer to reference sequence',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--limit', default=None, type=TypeOrNone(Positive(int)),
    help='Limit number of reads to process.')
parser.add_argument('--section', default='template', choices=['template', 'complement'],
    help='Section to call')
parser.add_argument('--strand_list', default=None, action=FileExist,
    help='strand summary file containing subset.')
parser.add_argument('--trim', default=(500, 50), nargs=2, type=Positive(int),
    metavar=('beginning', 'end'), help='Number of events to trim off start and end')
parser.add_argument('--version', nargs=0, action=display_version_and_exit,
    help='Display version information.')
parser.add_argument('--window', default=3, type=Positive(int), metavar='length',
    help='Window length for input features')
parser.add_argument('model', action=FileExist, help='Pickled model file')
parser.add_argument('input_folder', action=FileExist,
    help='Directory containing single-read fast5 files.')

def map_transducer(args, fn):
    _, kmer_to_state = seq_tools.all_kmers(length=1, rev_map=True)
    with fast5(fn) as f5:
        ev, _ = f5.get_any_mapping_data(args.section)
        name, seq = f5.get_reference_fasta(section=args.section).split()
        sn = f5.filename_short
    if len(ev) <= sum(args.trim):
        return None

    inMat = features.from_events(ev)[args.trim[0] : -args.trim[1]]
    inMat = np.expand_dims(inMat, axis=1)

    with open(args.model, 'r') as fh:
        calc_post = cPickle.load(fh)

    trans = np.squeeze(calc_post(inMat))
    seq = np.array(map(lambda k: kmer_to_state[k], seq))
    score, path = transducer.map_to_sequence(trans, seq, log=False)

    lb = args.trim[0] + 1
    ub = args.trim[1] + 1
    return sn, score, path, ev[lb:-ub]

if __name__ == '__main__':
    args = parser.parse_args()

    files = iterate_fast5(args.input_folder, paths=True, limit=args.limit, strand_list=args.strand_list)
    nbases = nevents = 0
    t0 = time.time()
    for res in tang_imap(map_transducer, files, threads=1, fix_args=[args], unordered=True):
        if res is None:
            continue
        read, score, path, ev = res
        ev = nprf.append_fields(ev, 'rnn_pos', path)
        tangio.numpy_savetsv( read + '.tsv', ev, header=True)
    dt = time.time() - t0
