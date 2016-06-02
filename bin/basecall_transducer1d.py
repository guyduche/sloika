#!/usr/bin/env python
import argparse
from six.moves import cPickle
import sys
import time

from dragonet.bio import seq_tools

from tang import numpty as np
from tang.fast5 import iterate_fast5, fast5
from tang.nn import features
from tang.util.cmdargs import (AutoBool, display_version_and_exit, FileExist,
                               probability, Positive, TypeOrNone, Vector)
from tang.util.tang_iter import tang_imap

# This is here, not in main to allow documentation to be built
parser = argparse.ArgumentParser(
    description='1D and 2D basecaller for Tang NN library',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#parser.add_argument('--jobs', default=1, type=int, action=CheckCPU,
#    help='Number of jobs to run in parallel.')
parser.add_argument('--individual', default=False, type=AutoBool,
    help='Return individual basecalls for each section')
parser.add_argument('--limit', default=None, type=TypeOrNone(Positive(int)),
    help='Limit number of reads to process.')
parser.add_argument('--min_prob', metavar='probability', default=1e-5,
    type=probability, help='Minimum allowed probabiility for basecalls')
parser.add_argument('--section', default='template', choices=['template', 'complement'],
    help='Section to call')
parser.add_argument('--strand_list', default=None, action=FileExist,
    help='strand summary file containing subset.')
parser.add_argument('--trans', default=None, action=Vector(probability), nargs=3,
    metavar=('stay', 'step', 'skip'), help='Base transition probabilities')
parser.add_argument('--trim', default=(500, 50), nargs=2, type=Positive(int),
    metavar=('beginning', 'end'), help='Number of events to trim off start and end')
parser.add_argument('--version', nargs=0, action=display_version_and_exit,
    help='Display version information.')
parser.add_argument('--window', default=3, type=Positive(int), metavar='length',
    help='Window length for input features')
parser.add_argument('model', action=FileExist, help='Pickled model file')
parser.add_argument('input_folder', action=FileExist,
    help='Directory containing single-read fast5 files.')

_ETA = 1e-300

def prepare_post(post, min_prob=1e-5, init_trans=None):
    post = np.squeeze(post, axis=1)
    bad_state = post.shape[1] - 1
    max_call = np.argmax(post, axis=1)
    post = post[max_call != bad_state]
    post = post[:,:-1]
    post /= _ETA + np.sum(post, axis=1).reshape((-1, 1))

    return min_prob + (1.0 - min_prob) * post

def basecall(args, fn):
    with fast5(fn) as f5:
        ev = f5.get_section_events(args.section)
        sn = f5.filename_short
    if len(ev) <= sum(args.trim):
        return None

    inMat = features.from_events(ev)[args.trim[0] : -args.trim[1]]
    inMat = np.expand_dims(inMat, axis=1)

    with open(args.model, 'r') as fh:
        calc_post = cPickle.load(fh)

    post = prepare_post(calc_post(inMat), args.min_prob)

    stay_state = post.shape[1] - 1
    call = np.argmax(post, axis=1)
    call = call[call != stay_state]
    score = np.sum(np.log(np.amax(post, axis=1)))

    return sn, score, call, inMat.shape[0]

class SeqPrinter(object):
    def __init__(self, kmerlen, fh=None):
        self.kmers = seq_tools.all_kmers(length=1)
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
        seq = ''.join(kmer_path)
        self.fh.write(">{} {} {} events to {} bases\n".format(read_name, score, nev, len(seq)))
        self.fh.write(seq + '\n')
        return len(seq)


if __name__ == '__main__':
    args = parser.parse_args()
    seq_printer = SeqPrinter(3)

    files = iterate_fast5(args.input_folder, paths=True, limit=args.limit, strand_list=args.strand_list)
    nbases = nevents = 0
    t0 = time.time()
    for res in tang_imap(basecall, files, threads=1, fix_args=[args], unordered=True):
        if res is None:
            continue
        read, score, call, nev = res
        seq_len = seq_printer.write(read, score, call, nev)
        nbases += seq_len
        nevents += nev
    dt = time.time() - t0
    sys.stderr.write('Called {} bases in {:.1f} s ({:.1f} bases/s or {:.1f} events/s)\n'.format(nbases, dt, nbases / dt, nevents / dt))
