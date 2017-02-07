from __future__ import print_function

import argparse
import cPickle
import numpy as np
import posixpath
import time
import os

from untangled import bio
from untangled.cmdargs import (AutoBool, FileExists, Maybe, NonNegative,
                               proportion, Positive, Vector, FileAbsent)
from untangled import fast5
from untangled.iterators import imap_mp

from sloika import helpers, batch, util


def create_output_strand_file(output_strand_list_entries, output_file_name):
    output_strand_list_entries.sort()

    with open(output_file_name, "w") as sl:
        sl.write('\t'.join(['filename', 'nev', 'score', 'nstay', 'seqlen', 'start', 'end']) + '\n')
        for strand_data in output_strand_list_entries:
            sl.write('\t'.join(map(lambda x: str(x), strand_data)) + '\n')


def chunkify_with_remap_main(argv, parser):
    parser.add_argument('--compile', default=None, type=Maybe(str),
                        help='File output compiled model')
    parser.add_argument('--min-prob', metavar='proportion', default=1e-5,
                        type=proportion, help='Minimum allowed probabiility for basecalls')
    parser.add_argument('--prior', nargs=2, metavar=('start', 'end'), default=(25.0, 25.0),
                        type=Maybe(NonNegative(float)), help='Mean of start and end positions')
    parser.add_argument('--slip', default=5.0, type=Maybe(NonNegative(float)),
                        help='Slip penalty')
    parser.add_argument('--transducer', default=True, action=AutoBool,
                        help='Model is transducer')
    parser.add_argument('--output-strand-list', default="strand_output_list.txt", action=FileAbsent,
                        help='strand summary output file')
    parser.add_argument('model', action=FileExists, help='Pickled model file')
    parser.add_argument('references', action=FileExists,
                        help='Reference sequences in fasta format')
    parser.add_argument('input_folder', action=FileExists,
                        help='Directory containing single-read fast5 files')
    parser.add_argument('output', action=FileAbsent, help='Output HDF5 file')

    args = parser.parse_args(argv)

    fast5_files = fast5.iterate_fast5(args.input_folder, paths=True, limit=args.limit,
                                      strand_list=args.input_strand_list)

    print('* Processing data using', args.threads, 'threads')

    kwarg_names = ['trim', 'min_prob', 'transducer', 'kmer_len', 'prior', 'slip', 'chunk_len', 'use_scaled', 'normalise']
    i = 0
    compiled_file = helpers.compile_model(args.model, args.compile)
    output_strand_list_entries = []
    bad_list = []
    chunk_list = []
    label_list = []
    for res in imap_mp(batch.chunk_remap_worker, fast5_files, threads=args.threads, fix_kwargs=util.get_kwargs(args,kwarg_names),
                       unordered=True, init=batch.init_chunk_remap_worker, initargs=[compiled_file, args.references, args.kmer_len]):
        if res is not None:
            i = util.progress_report(i)
            read, score, nev, path, seq, chunks, labels, bad_ev = res
            chunk_list.append(chunks)
            label_list.append(labels)
            bad_list.append(bad_ev)
            output_strand_list_entries.append([read, nev, -score / nev, np.sum(np.ediff1d(path, to_begin=1) == 0),
                                 len(seq), min(path), max(path)])

    print('\n* Creating HDF5 file')
    util.create_hdf5(args, chunk_list, label_list, bad_list)

    print('\n* Creating output strand file')
    create_output_strand_file(output_strand_list_entries, args.output_strand_list)

    if compiled_file != args.compile:
        os.remove(compiled_file)
