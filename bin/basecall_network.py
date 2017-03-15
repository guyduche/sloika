#!/usr/bin/env python
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import *

import argparse
import numpy as np
import os
import pickle
import sys
import time

from untangled import bio, fast5
from untangled.cmdargs import (AutoBool, FileAbsent, FileExists, Maybe,
                               NonNegative, proportion, Positive, Vector)
from untangled.iterators import imap_mp
from untangled.maths import mad

from sloika import helpers
from sloika.variables import nstate


# create the top-level parser
parser = argparse.ArgumentParser(
    description='1D basecaller for RNNs',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)


# common command line arguments to all subcommands
common_parser = argparse.ArgumentParser(add_help=False)
common_parser.add_argument('--compile', default=None, action=FileAbsent,
                           help='File output compiled model')
common_parser.add_argument('--input_strand_list', default=None, action=FileExists,
                           help='Strand summary file containing subset')
common_parser.add_argument('--jobs', default=1, metavar='n', type=Positive(int),
                           help='Number of threads to use when processing data')
common_parser.add_argument('--kmer_len', default=5, metavar='length', type=Positive(int),
                           help='Length of kmer')
common_parser.add_argument('--limit', default=None, metavar='reads',
                           type=Maybe(Positive(int)), help='Limit number of reads to process')
common_parser.add_argument('--min_prob', metavar='proportion', default=1e-5,
                           type=proportion, help='Minimum allowed probabiility for basecalls')
common_parser.add_argument('--skip', default=0.0,
                           type=Positive(float), help='Skip penalty')
common_parser.add_argument('--trans', default=None, action=Vector(proportion), nargs=3,
                           metavar=('stay', 'step', 'skip'), help='Base transition probabilities')
common_parser.add_argument('--transducer', default=True, action=AutoBool,
                           help='Model is transducer')

common_parser.add_argument('model', action=FileExists, help='Pickled model file')
common_parser.add_argument('input_folder', action=FileExists,
                           help='Directory containing single-read fast5 files')


# add subparsers for each command
subparsers = parser.add_subparsers(help='command', dest='command')
subparsers.required = True

parser_ev = subparsers.add_parser('events', parents=[common_parser], help='basecall from events',
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser_ev.add_argument('--bad', default=True, action=AutoBool,
                       help='Model emits bad events as a separate state')
parser_ev.add_argument('--section', default='template', choices=['template', 'complement'],
                       help='Section to call')
parser_ev.add_argument('--segmentation', default=fast5.__default_segmentation_analysis__,
                       metavar='location', help='Location of segmentation information')
parser_ev.add_argument('--trim', default=(50, 1), nargs=2, type=NonNegative(int),
                       metavar=('beginning', 'end'), help='Number of events to trim off start and end')
parser_ev.set_defaults(datatype='events')


parser_raw = subparsers.add_parser('raw', parents=[common_parser], help='basecall from raw signal',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser_raw.add_argument('--bad', default=True, action=AutoBool,
                        help='Model emits bad signal blocks as a separate state')
parser_raw.add_argument('--open_pore_fraction', metavar='proportion', default=0,
                        type=proportion, help='Max fraction of signal to trim due to open pore')
parser_raw.add_argument('--trim', default=(200, 10), nargs=2, type=NonNegative(int),
                        metavar=('beginning', 'end'), help='Number of samples to trim off start and end')
parser_raw.set_defaults(datatype='samples')


def init_worker(model):
    import pickle
    global calc_post
    with open(model, 'rb') as fh:
        calc_post = pickle.load(fh)


def trim_array(x, trim):
    begin = trim[0]
    end = None if trim[1] == 0 else -trim[1]
    return x[begin:end]


def decode_post(post, args, eta=1e-10):
    from sloika import decode
    assert post.shape[2] == nstate(args.kmer_len, transducer=args.transducer, bad_state=args.bad)
    post = decode.prepare_post(post, min_prob=args.min_prob, drop_bad=args.bad and not args.transducer)
    if args.transducer:
        score, call = decode.viterbi(post, args.kmer_len, skip_pen=args.skip)
    else:
        trans = olddecode.estimate_transitions(post, trans=args.trans)
        score, call = olddecode.decode_profile(post, trans=np.log(eta + trans), log=False)
    return score, call


def basecall_events(args, fn):
    from sloika import features
    try:
        with fast5.Reader(fn) as f5:
            ev = f5.get_section_events(args.section, analysis=args.segmentation)
            sn = f5.filename_short
    except Exception as e:
        sys.stderr.write("Error getting events for section {!r} in file {}\n{!r}\n".format(args.section, fn, e))
        return None

    ev = trim_array(ev, args.trim)
    if ev.size == 0:
        sys.stderr.write("Read too short in file {}\n".format(fn))
        return None

    inMat = features.from_events(ev, tag='')[:, None, :]
    score, call = decode_post(calc_post(inMat), args)

    return sn, score, call, inMat.shape[0]


def basecall_raw(args, fn):
    from sloika import batch, config
    try:
        with fast5.Reader(fn) as f5:
            signal = f5.get_read(raw=True)
            sn = f5.filename_short
    except Exception as e:
        sys.stderr.write("Error getting raw data for file {}\n{!r}\n".format(fn, e))
        return None

    signal = batch.trim_open_pore(signal, args.open_pore_fraction)
    signal = trim_array(signal, args.trim)
    if signal.size == 0:
        sys.stderr.write("Read too short in file {}\n".format(fn))
        return None

    inMat = (signal - np.median(signal)) / mad(signal)
    inMat = inMat[:, None, None].astype(config.sloika_dtype)
    score, call = decode_post(calc_post(inMat), args)

    return sn, score, call, inMat.shape[0]


class SeqPrinter(object):

    def __init__(self, kmerlen, datatype="events", transducer=False, fname=None):
        self.kmers = bio.all_kmers(kmerlen)
        self.transducer = transducer
        self.datatype = datatype

        if fname is None:
            self.fh = sys.stdout
            self.close_fh = False
        else:
            self.fh = open(fname, 'w')
            self.close_fh = True

    def __del__(self):
        if self.close_fh:
            self.fh.close()

    def write(self, read_name, score, call, nev):
        kmer_path = [self.kmers[i] for i in call]
        seq = bio.kmers_to_sequence(kmer_path, always_move=self.transducer)
        self.fh.write(">{} score {:.0f}, {} {} to {} bases\n".format(read_name, score,
                                                                     nev, self.datatype, len(seq)))
        self.fh.write(seq + '\n')
        return len(seq)


if __name__ == '__main__':
    args = parser.parse_args()

    assert args.command in ["events", "raw"]

    compiled_file = helpers.compile_model(args.model, args.compile)

    seq_printer = SeqPrinter(args.kmer_len, datatype=args.datatype,
                             transducer=args.transducer)

    files = fast5.iterate_fast5(args.input_folder, paths=True, limit=args.limit,
                                strand_list=args.input_strand_list)
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
    t = 'Called {} bases in {:.1f} s ({:.1f} bases/s or {:.1f} {}/s)\n'
    sys.stderr.write(t.format(nbases, dt, nbases / dt, nevents / dt, args.datatype))

    if compiled_file != args.compile:
        os.remove(compiled_file)
