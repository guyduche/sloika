#!/usr/bin/env python
import argparse
import cPickle
import numpy as np
import os
import sys
import time

from sloika import helpers
from sloika.variables import nstate

from untangled import bio
from untangled.cmdargs import (AutoBool, FileExists, Maybe, NonNegative,
                               proportion, Positive, Vector)
from untangled import fast5
from untangled.iterators import imap_mp


# This is here, not in main to allow documentation to be built
parser = argparse.ArgumentParser(
    description='1D basecaller for RNNs',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--bad', default=True, action=AutoBool,
    help='Model emits bad events as a separate state')
parser.add_argument('--compile', default=None, type=Maybe(str),
    help='File output compiled model')
parser.add_argument('--jobs', default=4, metavar='n', type=Positive(int),
    help='Number of jobs to run in parallel')
parser.add_argument('--kmer', default=5, metavar='length', type=Positive(int),
    help='Length of kmer')
parser.add_argument('--limit', default=None, metavar='reads',
    type=Maybe(Positive(int)), help='Limit number of reads to process.')
parser.add_argument('--min_prob', metavar='proportion', default=1e-5,
    type=proportion, help='Minimum allowed probabiility for basecalls')
parser.add_argument('--section', default='template', choices=['template', 'complement'],
    help='Section to call')
parser.add_argument('--skip', default=0.0, type=Positive(float), help='Skip penalty')
parser.add_argument('--strand_list', default=None, action=FileExists,
    help='strand summary file containing subset.')
parser.add_argument('--transducer', default=True, action=AutoBool,
    help='Model is transducer')
parser.add_argument('--trans', default=None, action=Vector(proportion), nargs=3,
    metavar=('stay', 'step', 'skip'), help='Base transition probabilities')
parser.add_argument('--trim', default=(50, 1), nargs=2, type=NonNegative(int),
    metavar=('beginning', 'end'), help='Number of events to trim off start and end')
parser.add_argument('model', action=FileExists, help='Pickled model file')
parser.add_argument('input_folder', action=FileExists,
    help='Directory containing single-read fast5 files.')

_ETA = 1e-10


def init_worker(model):
    import cPickle
    global calc_post
    with open(model, 'r') as fh:
        calc_post = cPickle.load(fh)


def basecall(args, fn):
    from sloika import decode, features, olddecode
    try:
        with fast5.Reader(fn) as f5:
            ev = f5.get_section_events(args.section)
            sn = f5.filename_short
    except:
        return None

    if len(ev) <= sum(args.trim):
        return None
    begin, end = args.trim
    end = None if end is 0 else -end
    ev = ev[begin : end]

    inMat = features.from_events(ev, tag='')
    inMat = np.expand_dims(inMat, axis=1)

    post = calc_post(inMat)
    assert post.shape[2] == nstate(args.kmer, transducer=args.transducer, bad_state=args.bad)
    post = decode.prepare_post(post, min_prob=args.min_prob,
                               drop_bad=args.bad and not args.transducer)

    if args.transducer:
        score, call = decode.viterbi(post, args.kmer, skip_pen=args.skip)
    else:
        trans = olddecode.estimate_transitions(post, trans=args.trans)
        score, call = olddecode.decode_profile(post, trans=np.log(_ETA + trans), log=False)

    return sn, score, call, inMat.shape[0]


class SeqPrinter(object):
    def __init__(self, kmerlen, transducer=False, fh=None):
        self.kmers = bio.all_kmers(kmerlen)
        self.transducer = transducer
        self.close_fh = False

        if fh is None:
            self.fh = sys.stdout
        else:
            if isinstance(fh, file):
                self.fh = fh
            else:
                self.fh = open(fh, 'w')
                self.close_fh = True

    def __del__(self):
        if self.close_fh:
            self.fh.close()

    def write(self, read_name, score, call, nev):
        kmer_path = [self.kmers[i] for i in call]
        seq = bio.kmers_to_sequence(kmer_path, always_move=self.transducer)
        self.fh.write(">{} {} {} events to {} bases\n".format(read_name, score,
                                                              nev, len(seq)))
        self.fh.write(seq + '\n')
        return len(seq)


if __name__ == '__main__':
    args = parser.parse_args()

    compiled_file = helpers.compile_model(args.model)
    if args.compile is not None:
        os.rename(compiled_file, args.compile)
        compiled_file = args.compile


    seq_printer = SeqPrinter(args.kmer, transducer=args.transducer)

    files = fast5.iterate_fast5(args.input_folder, paths=True, limit=args.limit,
                                strand_list=args.strand_list)
    nbases = nevents = 0
    t0 = time.time()
    for res in imap_mp(basecall, files, threads=args.jobs, fix_args=[args],
                       unordered=True, init=init_worker, initargs=[compiled_file]):
        if res is None:
            continue
        read, score, call, nev = res
        seq_len = seq_printer.write(read, score, call, nev)
        nbases += seq_len
        nevents += nev
    dt = time.time() - t0
    sys.stderr.write('Called {} bases in {:.1f} s ({:.1f} bases/s or {:.1f} events/s)\n'.format(nbases, dt, nbases / dt, nevents / dt))
