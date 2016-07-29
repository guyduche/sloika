#!/usr/bin/env python
import argparse
import cPickle
import sys
import time

from untangled.cmdargs import (display_version_and_exit, FileExists,
                               Maybe, proportion, Positive, Vector)
from untangled import fast5

from sloika import batch, __version__

# This is here, not in main to allow documentation to be built
parser = argparse.ArgumentParser(
    description='Calculate posteriors for template or complement strands',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch', default=1000, metavar='size', type=Positive(int),
    help='Batch size (number of chunks to run in parallel)')
parser.add_argument('--chunk', default=100, metavar='events', type=Positive(int),
    help='Length of each read chunk')
parser.add_argument('--limit', default=None, metavar='reads', type=Maybe(Positive(int)),
    help='Limit number of reads to process.')
parser.add_argument('--min_prob', metavar='proportion', default=1e-5,
    type=proportion, help='Minimum allowed probabiility for basecalls')
parser.add_argument('--section', default='template', choices=['template', 'complement'],
    help='Section to call')
parser.add_argument('--strand_list', default=None, action=FileExists,
    help='strand summary file containing subset.')
parser.add_argument('--trans', default=None, action=Vector(proportion), nargs=3,
    metavar=('stay', 'step', 'skip'), help='Base transition probabilities')
parser.add_argument('--trim', default=(500, 50), nargs=2, type=Positive(int),
    metavar=('beginning', 'end'), help='Number of events to trim off start and end')
parser.add_argument('--version', nargs=0, action=display_version_and_exit, metavar=__version__,
    help='Display version information.')
parser.add_argument('--window', default=3, type=Positive(int), metavar='length',
    help='Window length for input features')
parser.add_argument('model', action=FileExists, help='Pickled model file')
parser.add_argument('input_folder', action=FileExists,
    help='Directory containing single-read fast5 files.')



if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.model, 'r') as fh:
        calc_post = cPickle.load(fh)

    files = list(fast5.iterate_fast5(args.input_folder, paths=True,
                                     limit=args.limit,
                                     strand_list=args.strand_list))
    nevents = 0
    t0 = time.time()
    for in_data in batch.kmer(files, args.section, args.batch, args.chunk,
                              args.window, trim=args.trim,
                              use_scaled=args.use_scaled, kmer_len=args.kmer):
        post = calc_post(in_data[0])
        nevents += in_data.size
    dt = time.time() - t0
    sys.stderr.write('{} events in {:.1f} s ({:.1f} events/s)\n'.format(nevents, dt, nevents / dt))
