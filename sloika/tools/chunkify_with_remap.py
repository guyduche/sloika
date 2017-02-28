from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import *

import argparse
import pickle
import os
import posixpath
import sys
import time
import numpy as np

from untangled import bio
from untangled.cmdargs import (AutoBool, FileExists, Maybe, NonNegative,
                               proportion, Positive, Vector, FileAbsent)
from untangled import fast5
from untangled.iterators import imap_mp

from sloika import helpers, batch, util


def create_output_strand_file(output_strand_list_entries, output_file_name):
    output_strand_list_entries.sort()

    with open(output_file_name, "w") as sl:
        sl.write(u'\t'.join(['filename', 'nev', 'score', 'nstay', 'seqlen', 'start', 'end']) + u'\n')
        for strand_data in output_strand_list_entries:
            sl.write('\t'.join([str(x) for x in strand_data]) + '\n')


def chunkify_with_remap_main(argv, parser):
    parser.add_argument('--compile', default=None, type=Maybe(str),
                        help='File output compiled model')
    parser.add_argument('--min_prob', metavar='proportion', default=1e-5,
                        type=proportion, help='Minimum allowed probabiility for basecalls')
    parser.add_argument('--output_strand_list', default="strand_output_list.txt",
                        help='strand summary output file')
    parser.add_argument('--prior', nargs=2, metavar=('start', 'end'), default=(25.0, 25.0),
                        type=Maybe(NonNegative(float)), help='Mean of start and end positions')
    parser.add_argument('--slip', default=5.0, type=Maybe(NonNegative(float)),
                        help='Slip penalty')
    parser.add_argument('--transducer', default=True, action=AutoBool,
                        help='Model is transducer')

    parser.add_argument('model', action=FileExists, help='Pickled model file')
    parser.add_argument('references', action=FileExists,
                        help='Reference sequences in fasta format')

    args = parser.parse_args(argv)

    if not args.overwrite:
        if os.path.exists(args.output):
            print("Cowardly refusing to overwrite {}".format(args.output))
            sys.exit(1)
        if os.path.exists(args.output_strand_list):
            print("Cowardly refusing to overwrite {}".format(args.output_strand_list))
            sys.exit(2)

    fast5_files = fast5.iterate_fast5(args.input_folder, paths=True, limit=args.limit,
                                      strand_list=args.input_strand_list)

    print('* Processing data using', args.jobs, 'threads')

    kwarg_names = ['trim', 'min_prob', 'transducer', 'kmer_len', 'min_length',
                   'prior', 'slip', 'chunk_len', 'use_scaled', 'normalisation']
    i = 0
    compiled_file = helpers.compile_model(args.model, args.compile)
    output_strand_list_entries = []
    bad_list = []
    chunk_list = []
    label_list = []
    for res in imap_mp(batch.chunk_remap_worker, fast5_files, threads=args.jobs,
                       fix_kwargs=util.get_kwargs(args, kwarg_names),
                       unordered=True, init=batch.init_chunk_remap_worker,
                       initargs=[compiled_file, args.references, args.kmer_len]):
        if res is not None:
            i = util.progress_report(i)
            read, score, nev, path, seq, chunks, labels, bad_ev = res
            chunk_list.append(chunks)
            label_list.append(labels)
            bad_list.append(bad_ev)
            output_strand_list_entries.append([read, nev, -score / nev,
                                               np.sum(np.ediff1d(path, to_begin=1) == 0),
                                               len(seq), min(path), max(path)])

    if compiled_file != args.compile:
        os.remove(compiled_file)

    if chunk_list == []:
        print("no chunks were produced", file=sys.stderr)
        sys.exit(1)
    else:
        print('\n* Creating HDF5 file')
        util.create_hdf5(args, chunk_list, label_list, bad_list)

        print('\n* Creating output strand file')
        create_output_strand_file(output_strand_list_entries, args.output_strand_list)
